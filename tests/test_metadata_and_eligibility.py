import copy
import unittest
from datetime import timedelta

import shapely.geometry

from cross_sensor.config import EligibilityConfig
from cross_sensor.discovery.observation_discovery import Sensor, parse_observation
from cross_sensor.discovery.sentinel2_metadata import parse_boa_scaling, parse_granule_geometry
from cross_sensor.eligibility.comparison_eligibility import ComparisonEligibility, ReasonCode
from cross_sensor.eligibility.observation_pair_selector import ObservationPairSelector, deduplicate_acquisitions
from cross_sensor.errors import MetadataError
from cross_sensor.preprocessing.band_mapping import LANDSAT_C2_L2, SENTINEL2_L2A
from cross_sensor.synthetic import ACQUISITION, LANDSAT_ACQUISITION, SYNTHETIC_BBOX, SyntheticAssetSource, landsat_item, sentinel2_item

ELIGIBILITY = EligibilityConfig(
    sentinel2_platforms=("Sentinel-2A", "Sentinel-2B", "Sentinel-2C"),
    landsat_platforms=("landsat-8", "landsat-9"),
    landsat_collection_categories=("T1",),
    max_scene_cloud_cover_percent=70.0,
    max_time_difference_hours=72.0,
    min_valid_overlap_fraction=0.25,
)
AOI = shapely.geometry.box(*SYNTHETIC_BBOX)


class ParseObservationTest(unittest.TestCase):
    def test_parses_sentinel2_and_landsat(self) -> None:
        s2 = parse_observation(sentinel2_item("S2", ACQUISITION), Sensor.SENTINEL2)
        landsat = parse_observation(landsat_item("L8", LANDSAT_ACQUISITION), Sensor.LANDSAT)
        self.assertEqual(s2.grid_id, "43PGQ")
        self.assertEqual(landsat.grid_id, "144/051")
        self.assertAlmostEqual(landsat.sun_zenith_deg, 32.0)
        self.assertEqual(s2.acquired_at, ACQUISITION)

    def test_missing_or_invalid_fields_raise(self) -> None:
        base = sentinel2_item("S2", ACQUISITION)
        mutations = [
            lambda item: item["properties"].pop("datetime"),
            lambda item: item["properties"].update(datetime="27 Feb 2024"),
            lambda item: item["properties"].update(datetime="2024-02-27T05:07:51"),
            lambda item: item["properties"].pop("platform"),
            lambda item: item.pop("geometry"),
            lambda item: item["properties"].update({"eo:cloud_cover": "cloudy"}),
            lambda item: item["properties"].pop("s2:mgrs_tile"),
            lambda item: item.update(assets={}),
            lambda item: item["assets"]["B04"].pop("href"),
            lambda item: item.pop("id"),
        ]
        for mutate in mutations:
            item = copy.deepcopy(base)
            mutate(item)
            with self.subTest(mutation=mutate), self.assertRaises(MetadataError):
                parse_observation(item, Sensor.SENTINEL2)


class Sentinel2MetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SyntheticAssetSource()

    def test_boa_scaling(self) -> None:
        scaling = parse_boa_scaling(self.source.fetch_text("synthetic://S2/product-metadata"), ("B04", "B8A"))
        self.assertEqual(scaling["B04"].quantification_value, 10000.0)
        self.assertEqual(scaling["B8A"].add_offset, -1000.0)

    def test_pre_baseline_4_has_zero_offset(self) -> None:
        xml = '<root><BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE></root>'
        self.assertEqual(parse_boa_scaling(xml, ("B04",))["B04"].add_offset, 0.0)

    def test_partial_offsets_and_bad_xml_raise(self) -> None:
        with self.assertRaises(MetadataError):
            parse_boa_scaling('<r><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE><BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET></r>', ("B04",))
        with self.assertRaises(MetadataError):
            parse_boa_scaling("<r><unclosed></r>", ("B04",))
        with self.assertRaises(MetadataError):
            parse_boa_scaling("<r/>", ("B04",))

    def test_granule_geometry(self) -> None:
        geometry = parse_granule_geometry(self.source.fetch_text("synthetic://S2/granule-metadata"), ("B04", "B8A"))
        self.assertEqual(geometry.sun_zenith_deg, 32.0)
        self.assertEqual(geometry.view_azimuth_deg["B8A"], 103.0)
        self.assertEqual(geometry.sensing_time.date(), ACQUISITION.date())


class EligibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.eligibility = ComparisonEligibility(ELIGIBILITY, AOI, "EPSG:32643")

    def _decide(self, item: dict, sensor: Sensor) -> ReasonCode:
        mapping = SENTINEL2_L2A if sensor is Sensor.SENTINEL2 else LANDSAT_C2_L2
        return self.eligibility.evaluate(parse_observation(item, sensor), mapping).reason

    def test_reason_codes(self) -> None:
        self.assertIs(self._decide(sentinel2_item("ok", ACQUISITION), Sensor.SENTINEL2), ReasonCode.ACCEPTED)
        self.assertIs(self._decide(landsat_item("l7", LANDSAT_ACQUISITION, platform="landsat-7"), Sensor.LANDSAT), ReasonCode.UNSUPPORTED_PLATFORM)
        self.assertIs(self._decide(landsat_item("t2", LANDSAT_ACQUISITION, category="T2"), Sensor.LANDSAT), ReasonCode.UNSUPPORTED_COLLECTION_CATEGORY)
        self.assertIs(self._decide(sentinel2_item("cloud", ACQUISITION, cloud=90), Sensor.SENTINEL2), ReasonCode.HIGH_CLOUD_COVER)

        far = sentinel2_item("far", ACQUISITION)
        far["geometry"] = shapely.geometry.mapping(shapely.geometry.box(10, 10, 11, 11))
        self.assertIs(self._decide(far, Sensor.SENTINEL2), ReasonCode.OUTSIDE_AOI)

        partial = sentinel2_item("partial", ACQUISITION)
        partial["geometry"] = shapely.geometry.mapping(shapely.geometry.box(77.703, 12.9, 78.0, 13.0))
        self.assertIs(self._decide(partial, Sensor.SENTINEL2), ReasonCode.AOI_NOT_FULLY_COVERED)

        missing_band = sentinel2_item("noscl", ACQUISITION)
        missing_band["assets"].pop("SCL")
        self.assertIs(self._decide(missing_band, Sensor.SENTINEL2), ReasonCode.REQUIRED_BAND_MISSING)

        wrong_crs = landsat_item("crs", LANDSAT_ACQUISITION)
        wrong_crs["properties"]["proj:code"] = "EPSG:32644"
        self.assertIs(self._decide(wrong_crs, Sensor.LANDSAT), ReasonCode.INVALID_RASTER_METADATA)

        no_cloud = sentinel2_item("nocloud", ACQUISITION)
        no_cloud["properties"].pop("eo:cloud_cover")
        self.assertIs(self._decide(no_cloud, Sensor.SENTINEL2), ReasonCode.INVALID_RASTER_METADATA)


class PairSelectionTest(unittest.TestCase):
    def test_nearest_counterpart_and_threshold(self) -> None:
        s2 = [parse_observation(sentinel2_item(f"S2_{days}", ACQUISITION + timedelta(days=days)), Sensor.SENTINEL2) for days in (0, 5, 10)]
        landsat = [
            parse_observation(landsat_item("near", LANDSAT_ACQUISITION + timedelta(days=4)), Sensor.LANDSAT),
            parse_observation(landsat_item("far", LANDSAT_ACQUISITION + timedelta(days=30)), Sensor.LANDSAT),
        ]
        pairs = ObservationPairSelector(72.0).select(landsat, s2)
        self.assertEqual(pairs[0].sentinel2.item_id, "S2_5")
        self.assertIs(pairs[0].decision.reason, ReasonCode.ACCEPTED)
        self.assertIs(pairs[1].decision.reason, ReasonCode.EXCESSIVE_TIME_DIFFERENCE)
        self.assertAlmostEqual(pairs[0].signed_time_difference_hours, 24 - (3 + 4 / 60) / 60, places=3)

    def test_no_counterpart(self) -> None:
        landsat = [parse_observation(landsat_item("alone", LANDSAT_ACQUISITION), Sensor.LANDSAT)]
        self.assertIs(ObservationPairSelector(72.0).select(landsat, [])[0].decision.reason, ReasonCode.NO_COUNTERPART)

    def test_duplicates_keep_latest_processing(self) -> None:
        first = sentinel2_item("old", ACQUISITION)
        second = sentinel2_item("new", ACQUISITION)
        second["properties"]["s2:processing_baseline"] = "05.11"
        kept, rejected = deduplicate_acquisitions([parse_observation(first, Sensor.SENTINEL2), parse_observation(second, Sensor.SENTINEL2)])
        self.assertEqual([observation.item_id for observation in kept], ["new"])
        self.assertIs(rejected[0][1].reason, ReasonCode.DUPLICATE_ACQUISITION)


if __name__ == "__main__":
    unittest.main()
