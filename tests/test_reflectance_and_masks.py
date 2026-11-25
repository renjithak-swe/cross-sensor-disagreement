import unittest

import numpy as np

from cross_sensor.comparison.ndvi_comparison import compute_ndvi
from cross_sensor.errors import MetadataError
from cross_sensor.preprocessing.band_mapping import HLS_L30, LANDSAT_C2_L2, SENTINEL2_L2A, AuxiliaryRole, SpectralRole
from cross_sensor.preprocessing.quality_masking import (
    hls_high_aerosol,
    hls_invalid_from_fmask,
    landsat_invalid_from_qa,
    sentinel2_invalid_from_scl,
)
from cross_sensor.preprocessing.reflectance_scaling import (
    apply_scaling,
    file_scaling,
    landsat_scaling,
    sentinel2_scaling,
)

LANDSAT_BANDS = ({"scale": 2.75e-05, "offset": -0.2, "nodata": 0},)


class ReflectanceScalingTest(unittest.TestCase):
    def test_landsat_scale_and_offset_come_from_stac(self) -> None:
        scaling = landsat_scaling(LANDSAT_BANDS, "red")
        reflectance = apply_scaling(np.array([[10000, 20000]], dtype=np.uint16), scaling)
        np.testing.assert_allclose(reflectance, [[10000 * 2.75e-5 - 0.2, 20000 * 2.75e-5 - 0.2]], rtol=1e-6)

    def test_landsat_nodata_and_out_of_range_become_nan(self) -> None:
        scaling = landsat_scaling(LANDSAT_BANDS, "red")
        reflectance = apply_scaling(np.array([0, 7000, 50000, 12000], dtype=np.uint16), scaling)
        self.assertTrue(np.isnan(reflectance[:3]).all())
        self.assertFalse(np.isnan(reflectance[3]))

    def test_landsat_missing_scale_raises(self) -> None:
        with self.assertRaises(MetadataError):
            landsat_scaling(({"nodata": 0},), "red")
        with self.assertRaises(MetadataError):
            landsat_scaling((), "red")

    def test_sentinel2_offset_is_applied(self) -> None:
        scaling = sentinel2_scaling(10000.0, -1000.0, "B04")
        reflectance = apply_scaling(np.array([1000, 1500, 0, 65535], dtype=np.uint16), scaling)
        self.assertTrue(np.isnan(reflectance[0]), "DN 1000 is zero reflectance, which is non-physical for NDVI")
        self.assertAlmostEqual(float(reflectance[1]), 0.05, places=6)
        self.assertTrue(np.isnan(reflectance[2]), "DN 0 is nodata")
        self.assertTrue(np.isnan(reflectance[3]), "DN 65535 is saturated")

    def test_sentinel2_without_offset_matches_old_baselines(self) -> None:
        reflectance = apply_scaling(np.array([500], dtype=np.uint16), sentinel2_scaling(10000.0, 0.0, "B04"))
        self.assertAlmostEqual(float(reflectance[0]), 0.05, places=6)

    def test_file_scaling_requires_a_positive_scale(self) -> None:
        with self.assertRaises(MetadataError):
            file_scaling(None, 0.0, -9999, "x")
        scaling = file_scaling(0.0001, 0.0, -9999, "x")
        reflectance = apply_scaling(np.array([-9999, 2500], dtype=np.int16), scaling)
        self.assertTrue(np.isnan(reflectance[0]))
        self.assertAlmostEqual(float(reflectance[1]), 0.25, places=6)


class NdviTest(unittest.TestCase):
    def test_values(self) -> None:
        ndvi = compute_ndvi(np.array([0.1, 0.2], dtype=np.float32), np.array([0.3, 0.2], dtype=np.float32))
        np.testing.assert_allclose(ndvi, [0.5, 0.0], atol=1e-6)

    def test_undefined_values_are_nan(self) -> None:
        ndvi = compute_ndvi(np.array([0.0, np.nan, 0.1]), np.array([0.0, 0.3, np.nan]))
        self.assertTrue(np.isnan(ndvi).all())

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_ndvi(np.zeros(2), np.zeros(3))


class QualityMaskingTest(unittest.TestCase):
    def test_scl_keeps_only_listed_classes(self) -> None:
        invalid = sentinel2_invalid_from_scl(np.arange(12), [4, 5, 6])
        self.assertEqual(list(np.flatnonzero(~invalid)), [4, 5, 6])

    def test_landsat_qa_bits_and_saturation(self) -> None:
        clear = 1 << 6
        pixel_qa = np.array([clear, clear | 1 << 3, clear | 1 << 4, 1, clear | 1 << 7, clear])
        saturation = np.array([0, 0, 0, 0, 0, 1 << 3])
        invalid = landsat_invalid_from_qa(pixel_qa, saturation, [0, 1, 2, 3, 4, 5], [3, 4])
        self.assertEqual(list(invalid), [False, True, True, True, False, True])

    def test_landsat_qa_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            landsat_invalid_from_qa(np.zeros(2), np.zeros(3), [0], [3])

    def test_hls_fmask(self) -> None:
        fmask = np.array([0, 1 << 1, 1 << 2, 1 << 5, 0b11 << 6, 255], dtype=np.uint8)
        self.assertEqual(list(hls_invalid_from_fmask(fmask, [1, 2, 3, 4], False)), [False, True, True, False, False, True])
        self.assertEqual(list(hls_invalid_from_fmask(fmask, [1, 2, 3, 4], True)), [False, True, True, False, True, True])
        self.assertEqual(list(hls_high_aerosol(fmask)), [False, False, False, False, True, False])


class BandMappingTest(unittest.TestCase):
    def test_required_assets(self) -> None:
        self.assertEqual(SENTINEL2_L2A.asset_for(SpectralRole.NIR_NARROW), "B8A")
        self.assertEqual(LANDSAT_C2_L2.asset_for(SpectralRole.NIR_NARROW), "nir08")
        self.assertEqual(HLS_L30.asset_for(SpectralRole.NIR_NARROW), "B05")
        self.assertEqual(LANDSAT_C2_L2.missing_assets(["red", "nir08", "qa_pixel"]), ["qa_radsat"])

    def test_unknown_role_raises(self) -> None:
        with self.assertRaises(KeyError):
            LANDSAT_C2_L2.asset_for(SpectralRole.NIR_BROAD)
        with self.assertRaises(KeyError):
            LANDSAT_C2_L2.asset_for(AuxiliaryRole.FMASK)


if __name__ == "__main__":
    unittest.main()
