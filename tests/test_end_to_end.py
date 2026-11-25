import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas

from cross_sensor.comparison.comparison_run import Condition
from cross_sensor.provenance.hashing import sha256_file
from cross_sensor.runner import execute, pinned_ids_from_manifest, rerender_figures
from cross_sensor.synthetic import (
    SENTINEL2_GAIN,
    SyntheticAssetSource,
    SyntheticCatalog,
    run_synthetic_demo,
    surface_reflectance,
    synthetic_aoi_projected_bounds,
    synthetic_config,
)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls._directory.name) / "run"
        cls.manifest = run_synthetic_demo(cls.output, should_render_figures=True)
        cls.summary = json.loads((cls.output / "summary.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_every_output_is_hashed_in_the_manifest(self) -> None:
        recorded = {entry.path: entry.sha256 for entry in self.manifest.outputs}
        for name in ("observations.parquet", "pairs.parquet", "pair_metrics.parquet", "summary.json", "footprints.geoparquet"):
            self.assertEqual(recorded[name], sha256_file(self.output / name))
        self.assertTrue(any(path.startswith("rasters/") and path.endswith(".tif") for path in recorded))
        self.assertTrue(any(path.startswith("figures/") and path.endswith("-narrow.svg") for path in recorded))
        manifest_text = (self.output / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("sig=", manifest_text)

    def test_rejections_are_reported_not_dropped(self) -> None:
        observations = pandas.read_parquet(self.output / "observations.parquet")
        reasons = dict(zip(observations["item_id"], observations["reason"], strict=True))
        self.assertEqual(reasons["LE07_SYNTH_ETM"], "UNSUPPORTED_PLATFORM")
        self.assertEqual(reasons["S2A_SYNTH_CLOUDY"], "HIGH_CLOUD_COVER")
        pairs = pandas.read_parquet(self.output / "pairs.parquet")
        self.assertEqual(sorted(pairs["reason"]), ["ACCEPTED", "EXCESSIVE_TIME_DIFFERENCE"])
        self.assertRegex(pairs.set_index("pair_id").loc["P001", "common_mask_sha256"], r"^[0-9a-f]{64}$")

    def test_known_bias_is_recovered_on_the_common_grid(self) -> None:
        metrics = pandas.read_parquet(self.output / "pair_metrics.parquet")
        red = metrics[(metrics["condition"] == Condition.C1_AREA_WEIGHTED.value) & (metrics["variable"] == "red") & (metrics["landcover_class"] == 0)].iloc[0]
        # Linear field: the area-weighted Sentinel-2 mean equals the field at the Landsat cell centre, so the
        # residual is exactly (gain - 1) times the field, up to DN quantisation.
        left, bottom, right, top = synthetic_aoi_projected_bounds()
        expected = (SENTINEL2_GAIN - 1) * float(surface_reflectance("red", np.array((left + right) / 2), np.array((bottom + top) / 2)))
        self.assertAlmostEqual(red["mean_bias"], expected, delta=0.05 * expected)
        self.assertGreater(red["pearson_r"], 0.99)

    def test_all_conditions_ran(self) -> None:
        ran = set(self.summary["pooled"])
        self.assertTrue({condition.value for condition in Condition} <= ran, ran ^ {condition.value for condition in Condition})
        self.assertEqual(self.summary["ladder"]["pairs"], ["P001"])
        self.assertEqual(self.summary["hls_bandpass_tag_check"][0]["hls_tags"]["red"], [0.9765, 0.0009])

    def test_hls_products_match_landsat_exactly(self) -> None:
        f = self.summary["pooled"][Condition.F_HLS.value]["red"]
        self.assertLess(abs(f["mean_bias"]["estimate"]), 1e-4)

    def test_rerun_from_manifest_reproduces_the_metrics(self) -> None:
        recorded = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        pinned = pinned_ids_from_manifest(recorded)
        self.assertEqual(pinned["sentinel2"], ["S2A_SYNTH_PAIRED"])
        rerun_directory = Path(self._directory.name) / "rerun"
        config = synthetic_config(Path(self._directory.name) / "rerun-config")
        execute(config, SyntheticCatalog(), SyntheticAssetSource(), rerun_directory, pinned, should_render_figures=False)
        original = pandas.read_parquet(self.output / "pair_metrics.parquet")
        rerun = pandas.read_parquet(rerun_directory / "pair_metrics.parquet")
        pandas.testing.assert_frame_equal(original.reset_index(drop=True), rerun.reset_index(drop=True))

    def test_redrawn_figures_are_rehashed(self) -> None:
        manifest_path = rerender_figures(self.output)
        outputs = json.loads(manifest_path.read_text(encoding="utf-8"))["outputs"]
        figures = [entry for entry in outputs if entry["path"].startswith("figures/")]
        self.assertTrue(figures)
        for entry in outputs:
            self.assertEqual(entry["sha256"], sha256_file(self.output / entry["path"]), entry["path"])

    def test_native_condition_uses_the_same_ground_area(self) -> None:
        metrics = pandas.read_parquet(self.output / "pair_metrics.parquet")
        overall = metrics[(metrics["landcover_class"] == 0) & (metrics["variable"] == "ndvi")].set_index("condition")["count"]
        self.assertEqual(overall[Condition.A_NATIVE.value], overall[Condition.C1_AREA_WEIGHTED.value] * 36)
        self.assertTrue(np.isclose(overall[Condition.C2_BILINEAR.value], overall[Condition.C1_AREA_WEIGHTED.value]))


if __name__ == "__main__":
    unittest.main()


class TileTimeTest(EndToEndTest.__base__):
    def test_reported_gap_uses_tile_sensing_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_synthetic_demo(output, should_render_figures=False)
            pairs = pandas.read_parquet(output / "pairs.parquet").set_index("pair_id")
            # Synthetic granule SENSING_TIME is 05:07:51.5; Landsat is 05:10:55.
            self.assertAlmostEqual(pairs.loc["P001", "tile_signed_time_difference_hours"], -183.5 / 3600, places=6)
