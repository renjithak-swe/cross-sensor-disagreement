import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cross_sensor.config import load_config, parse_config
from cross_sensor.errors import ConfigError
from cross_sensor.provenance.experiment_manifest import (
    ExperimentManifest,
    build_experiment_id,
    load_manifest,
    strip_query_string,
)
from cross_sensor.provenance.hashing import sha256_file, sha256_json
from cross_sensor.synthetic import synthetic_config

PROJECT = Path(__file__).resolve().parent.parent


class ConfigTest(unittest.TestCase):
    def test_project_config_loads(self) -> None:
        config = load_config(PROJECT / "configs" / "experiment.toml")
        self.assertEqual(config.grid.crs, "EPSG:32643")
        self.assertTrue(config.aoi_path.exists())
        self.assertEqual(config.eligibility.max_time_difference_hours, 72.0)

    def test_invalid_values_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = synthetic_config(Path(directory)).raw
            mutations = [
                lambda value: value.pop("grid"),
                lambda value: value["grid"].update(subcell_m=7.0),
                lambda value: value["period"].update(end="2023-01-01"),
                lambda value: value["period"].update(start="yesterday"),
                lambda value: value["eligibility"].update(min_valid_overlap_fraction=1.5),
                lambda value: value["eligibility"].update(max_time_difference_hours=True),
                lambda value: value["eligibility"].update(landsat_platforms=[]),
                lambda value: value["quality"].update(landsat_invalid_qa_bits=["cloud"]),
                lambda value: value["uncertainty"].update(bootstrap_iterations=5),
            ]
            for mutate in mutations:
                broken = copy.deepcopy(raw)
                mutate(broken)
                with self.subTest(mutation=mutate), self.assertRaises(ConfigError):
                    parse_config(broken, Path(directory))


class ManifestTest(unittest.TestCase):
    def test_signature_is_stripped(self) -> None:
        self.assertEqual(strip_query_string("https://host/a/b.tif?st=1&sig=secret"), "https://host/a/b.tif")

    def test_experiment_id_depends_on_config(self) -> None:
        moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertNotEqual(build_experiment_id({"a": 1}, moment), build_experiment_id({"a": 2}, moment))
        self.assertEqual(sha256_json({"b": 1, "a": 2}), sha256_json({"a": 2, "b": 1}))

    def test_outputs_are_hashed_and_manifest_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "table.parquet"
            output.write_bytes(b"content")
            manifest = ExperimentManifest(
                experiment_id="x", executed_at="t", catalog_endpoint="e", collections={}, aoi={}, config={"k": 1},
                config_sha256="c", grid={}, resampling={}, quality_masks={}, software_versions={}, git_commit=None,
                source_observations=[],
            )
            manifest.record_outputs(root, [output])
            manifest.write(root / "manifest.json")
            loaded = load_manifest(root / "manifest.json")
            self.assertEqual(loaded["outputs"][0]["sha256"], sha256_file(output))
            self.assertEqual(loaded["outputs"][0]["path"], "table.parquet")

    def test_unknown_manifest_version_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"manifest_version": 99, "config": {}, "source_observations": []}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
