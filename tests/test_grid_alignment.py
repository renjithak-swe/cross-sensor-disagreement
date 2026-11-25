import unittest

import numpy as np
from affine import Affine

from cross_sensor.errors import GridAlignmentError
from cross_sensor.harmonization.grid_alignment import (
    GdalResampling,
    GridDefinition,
    Reducer,
    block_reduce,
    block_reduce_angle,
    build_aligned_grid,
    dilate,
    kernel_radius,
    replicate_to_grid,
    resample_with_gdal,
    with_resolution,
)

CRS = "EPSG:32643"


class AlignedGridTest(unittest.TestCase):
    def test_outward_grid_has_edges_at_offset(self) -> None:
        grid = build_aligned_grid((100.0, 200.0, 190.0, 260.0), CRS, 30.0, 15.0)
        self.assertEqual(grid.transform.c % 30, 15)
        self.assertEqual(grid.transform.f % 30, 15)
        left, bottom, right, top = grid.bounds
        self.assertTrue(left <= 100 and bottom <= 200 and right >= 190 and top >= 260)

    def test_inward_grid_stays_inside(self) -> None:
        outer = build_aligned_grid((100.0, 200.0, 190.0, 260.0), CRS, 30.0, 15.0)
        inner = build_aligned_grid(outer.bounds, CRS, 30.0, 0.0, is_inward=True)
        self.assertEqual(inner.transform.c % 30, 0)
        self.assertEqual((inner.width, inner.height), (outer.width - 1, outer.height - 1))

    def test_with_resolution_requires_whole_ratio(self) -> None:
        grid = build_aligned_grid((0.0, 0.0, 60.0, 60.0), CRS, 30.0, 0.0)
        self.assertEqual(with_resolution(grid, 5.0).shape, (12, 12))
        with self.assertRaises(GridAlignmentError):
            with_resolution(grid, 7.0)


class ExactMovesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coarse = GridDefinition(CRS, Affine(30, 0, 15, 0, -30, 75), 2, 2)
        self.fine = with_resolution(self.coarse, 5.0)

    def test_replication_then_mean_is_area_weighted(self) -> None:
        ten_metre = np.arange(64, dtype=np.float32).reshape(8, 8)
        transform = Affine(10, 0, 0, 0, -10, 80)
        fine = replicate_to_grid(ten_metre, transform, self.fine)
        mean = block_reduce(fine, self.fine, self.coarse, Reducer.MEAN)
        # The first 30 m cell spans x 15-45 and y 45-75: half of column 1, all of columns 2-3, half of column 4,
        # and the same split for rows 0.5-3.5.
        weights = np.array([0.5, 1.0, 1.0, 0.5]) / 3.0
        expected = (weights[:, None] * weights[None, :] * ten_metre[0:4, 1:5]).sum()
        self.assertAlmostEqual(float(mean[0, 0]), float(expected), places=4)

    def test_area_weighted_mean_matches_gdal_average(self) -> None:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        rng = np.random.default_rng(0)
        values = rng.uniform(0, 1, (40, 40)).astype(np.float32)
        transform = Affine(10, 0, 0, 0, -10, 400)
        coarse = GridDefinition(CRS, Affine(30, 0, 15, 0, -30, 375), 11, 11)
        ours = block_reduce(replicate_to_grid(values, transform, with_resolution(coarse, 5.0)), with_resolution(coarse, 5.0), coarse, Reducer.MEAN)
        gdal = np.zeros(coarse.shape, dtype=np.float32)
        reproject(values, gdal, src_transform=transform, src_crs=CRS, dst_transform=coarse.transform, dst_crs=CRS, resampling=Resampling.average)
        np.testing.assert_allclose(ours, gdal, atol=1e-5)

    def test_block_all_and_any(self) -> None:
        mask = np.zeros(self.fine.shape, dtype=bool)
        mask[0, 0] = True
        self.assertTrue(block_reduce(mask, self.fine, self.coarse, Reducer.ANY)[0, 0])
        self.assertFalse(block_reduce(mask, self.fine, self.coarse, Reducer.ALL)[0, 0])
        self.assertFalse(block_reduce(mask, self.fine, self.coarse, Reducer.ANY)[1, 1])

    def test_misaligned_source_raises(self) -> None:
        with self.assertRaises(GridAlignmentError):
            replicate_to_grid(np.zeros((8, 8)), Affine(10, 0, 2, 0, -10, 80), self.fine)

    def test_source_too_small_raises(self) -> None:
        with self.assertRaises(GridAlignmentError):
            replicate_to_grid(np.zeros((2, 2)), Affine(10, 0, 0, 0, -10, 80), self.fine)

    def test_circular_mean_of_azimuths(self) -> None:
        fine = np.where(np.arange(self.fine.width)[None, :] % 2 == 0, 359.0, 1.0) * np.ones(self.fine.shape)
        mean = block_reduce_angle(fine, self.fine, self.coarse)
        self.assertTrue(np.all(np.minimum(mean, 360 - mean) < 1e-3))


class GdalResamplingTest(unittest.TestCase):
    def test_kernel_radius_widens_when_downsampling(self) -> None:
        self.assertEqual(kernel_radius(GdalResampling.NEAREST, 10, 30), 0)
        self.assertEqual(kernel_radius(GdalResampling.BILINEAR, 10, 30), 3)
        self.assertEqual(kernel_radius(GdalResampling.CUBIC, 30, 30), 2)

    def test_dilate(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        self.assertEqual(int(dilate(mask, 1).sum()), 9)
        self.assertEqual(int(dilate(mask, 0).sum()), 1)

    def test_invalid_pixels_never_leak_into_valid_cells(self) -> None:
        values = np.full((30, 30), 0.2, dtype=np.float32)
        invalid = np.zeros(values.shape, dtype=bool)
        invalid[10:12, 10:12] = True
        values[invalid] = 999.0
        grid = GridDefinition(CRS, Affine(30, 0, 15, 0, -30, 285), 9, 9)
        resampled, resampled_invalid = resample_with_gdal(values, Affine(10, 0, 0, 0, -10, 300), CRS, grid, GdalResampling.BILINEAR, invalid)
        self.assertTrue(resampled_invalid.any())
        np.testing.assert_allclose(resampled[~resampled_invalid], 0.2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()


class CoarsenOntoTest(unittest.TestCase):
    def test_ten_metre_pixels_average_onto_offset_twenty_metre_grid(self) -> None:
        from cross_sensor.harmonization.grid_alignment import coarsen_onto

        values = np.arange(36, dtype=np.float32).reshape(6, 6)
        source = Affine(10, 0, 0, 0, -10, 60)
        # Target starts one 10 m pixel left of the source, so its first column half-overlaps nothing.
        result = coarsen_onto(values, source, Affine(20, 0, -10, 0, -20, 60), (3, 3))
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertAlmostEqual(float(result[0, 1]), float(values[0:2, 1:3].mean()))
        self.assertAlmostEqual(float(result[2, 2]), float(values[4:6, 3:5].mean()))

    def test_misaligned_target_raises(self) -> None:
        from cross_sensor.harmonization.grid_alignment import coarsen_onto

        with self.assertRaises(GridAlignmentError):
            coarsen_onto(np.zeros((4, 4)), Affine(10, 0, 0, 0, -10, 40), Affine(20, 0, 5, 0, -20, 40), (2, 2))
