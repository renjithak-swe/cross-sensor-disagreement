import math
import unittest

import numpy as np

from cross_sensor.errors import UnsupportedPlatformError
from cross_sensor.harmonization.geometry_normalization import c_factor, li_sparse_reciprocal_kernel, ross_thick_kernel
from cross_sensor.harmonization.spectral_adjustment import adjust_to_oli, coefficients_for, parse_hls_bandpass_tags
from cross_sensor.metrics.residual_analyzer import BIN_WIDTH, ResidualAccumulator, ResidualMoments, summarize_residuals
from cross_sensor.metrics.uncertainty import (
    bootstrap_paired_difference,
    bootstrap_pooled,
    bootstrap_spearman,
    mean_absolute_error,
    mean_bias,
)
from cross_sensor.preprocessing.band_mapping import SpectralRole


class ResidualTest(unittest.TestCase):
    def test_known_values(self) -> None:
        summary = summarize_residuals(np.array([0.1, 0.2, 0.3, np.nan]), np.array([0.2, 0.2, 0.1, 0.5]))
        self.assertEqual(summary.count, 3)
        self.assertAlmostEqual(summary.mean_bias, (0.1 + 0.0 - 0.2) / 3)
        self.assertAlmostEqual(summary.mean_absolute_error, 0.1)
        self.assertAlmostEqual(summary.median_absolute_error, 0.1)
        self.assertAlmostEqual(summary.root_mean_square_error, math.sqrt((0.01 + 0.04) / 3))

    def test_high_correlation_can_hide_a_bias(self) -> None:
        landsat = np.linspace(0.1, 0.8, 500)
        summary = summarize_residuals(landsat, landsat + 0.1)
        self.assertGreater(summary.pearson_r, 0.999)
        self.assertAlmostEqual(summary.mean_bias, 0.1, places=6)
        self.assertAlmostEqual(summary.mean_absolute_error, 0.1, places=6)

    def test_empty_input(self) -> None:
        self.assertEqual(summarize_residuals(np.array([np.nan]), np.array([1.0])).count, 0)

    def test_accumulator_matches_exact_statistics(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 0.8, 4000)
        y = x + rng.normal(0.02, 0.05, x.size)
        accumulator = ResidualAccumulator()
        accumulator.add(x[:1500], y[:1500])
        accumulator.add(x[1500:], y[1500:])
        pooled, exact = accumulator.summary(), summarize_residuals(x, y)
        self.assertEqual(pooled.count, exact.count)
        for field in ("mean_bias", "mean_absolute_error", "root_mean_square_error", "residual_standard_deviation", "pearson_r"):
            self.assertAlmostEqual(getattr(pooled, field), getattr(exact, field), places=9)
        self.assertLessEqual(abs(pooled.median_absolute_error - exact.median_absolute_error), BIN_WIDTH)


class BootstrapTest(unittest.TestCase):
    def _moments(self, offsets: list[float]) -> list[ResidualMoments]:
        landsat = np.linspace(0.2, 0.6, 100)
        return [ResidualMoments.from_arrays(landsat, landsat + offset) for offset in offsets]

    def test_pooled_interval_brackets_estimate_and_is_reproducible(self) -> None:
        moments = self._moments([0.01, 0.02, 0.03, 0.05, 0.04])
        first = bootstrap_pooled(moments, mean_absolute_error, 500, np.random.default_rng(3))
        second = bootstrap_pooled(moments, mean_absolute_error, 500, np.random.default_rng(3))
        self.assertAlmostEqual(first.estimate, 0.03)
        self.assertLessEqual(first.lower, first.estimate)
        self.assertGreaterEqual(first.upper, first.estimate)
        self.assertEqual(first, second)

    def test_single_pair_has_no_interval(self) -> None:
        interval = bootstrap_pooled(self._moments([0.02]), mean_bias, 200, np.random.default_rng(0))
        self.assertTrue(math.isnan(interval.lower))

    def test_paired_difference(self) -> None:
        before, after = self._moments([0.04, 0.05, 0.06]), self._moments([0.01, 0.02, 0.03])
        interval = bootstrap_paired_difference(before, after, mean_absolute_error, 500, np.random.default_rng(0))
        self.assertAlmostEqual(interval.estimate, -0.03)
        self.assertLess(interval.upper, 0)
        with self.assertRaises(ValueError):
            bootstrap_paired_difference(before, after[:2], mean_absolute_error, 10, np.random.default_rng(0))

    def test_spearman(self) -> None:
        x = np.arange(10, dtype=float)
        interval = bootstrap_spearman(x, x**2, 200, np.random.default_rng(0))
        self.assertAlmostEqual(interval.estimate, 1.0)
        self.assertTrue(math.isnan(bootstrap_spearman(x[:2], x[:2], 10, np.random.default_rng(0)).estimate))


class BrdfKernelTest(unittest.TestCase):
    def test_kernels_vanish_at_nadir_with_overhead_sun(self) -> None:
        zero = np.array([0.0])
        self.assertAlmostEqual(float(ross_thick_kernel(zero, zero, zero)[0]), 0.0, places=12)
        self.assertAlmostEqual(float(li_sparse_reciprocal_kernel(zero, zero, zero)[0]), 0.0, places=12)

    def test_kernels_are_reciprocal(self) -> None:
        sun, view, azimuth = np.deg2rad([35.0]), np.deg2rad([8.0]), np.deg2rad([60.0])
        for kernel in (ross_thick_kernel, li_sparse_reciprocal_kernel):
            self.assertAlmostEqual(float(kernel(sun, view, azimuth)[0]), float(kernel(view, sun, azimuth)[0]), places=12)

    def test_backscatter_is_brighter_than_forward_scatter(self) -> None:
        sun, view = np.deg2rad([30.0]), np.deg2rad([10.0])
        backward = li_sparse_reciprocal_kernel(sun, view, np.array([0.0]))
        forward = li_sparse_reciprocal_kernel(sun, view, np.array([np.pi]))
        self.assertGreater(float(backward[0]), float(forward[0]))

    def test_c_factor_is_one_at_target_geometry(self) -> None:
        factor = c_factor(SpectralRole.RED, np.array([30.0]), np.array([0.0]), np.array([120.0]), np.array([0.0]), np.array([30.0]))
        self.assertAlmostEqual(float(factor[0]), 1.0, places=6)

    def test_c_factor_lowers_backscatter_and_raises_forward_scatter(self) -> None:
        backward = c_factor(SpectralRole.NIR_NARROW, np.array([30.0]), np.array([7.0]), np.array([120.0]), np.array([120.0]), np.array([30.0]))
        forward = c_factor(SpectralRole.NIR_NARROW, np.array([30.0]), np.array([7.0]), np.array([120.0]), np.array([300.0]), np.array([30.0]))
        self.assertLess(float(backward[0]), 1.0)
        self.assertGreater(float(forward[0]), 1.0)


class BandpassTest(unittest.TestCase):
    def test_published_coefficients(self) -> None:
        coefficients = coefficients_for("Sentinel-2A", SpectralRole.RED)
        self.assertEqual((coefficients.slope, coefficients.intercept), (0.9765, 0.0009))
        adjusted = adjust_to_oli(np.array([0.1], dtype=np.float32), "Sentinel-2B", SpectralRole.NIR_NARROW)
        self.assertAlmostEqual(float(adjusted[0]), 0.09966, places=6)

    def test_unknown_platform_or_band_raises(self) -> None:
        with self.assertRaises(UnsupportedPlatformError):
            coefficients_for("Sentinel-2D", SpectralRole.RED)
        with self.assertRaises(UnsupportedPlatformError):
            coefficients_for("Sentinel-2A", SpectralRole.NIR_BROAD)

    def test_hls_tags_parse(self) -> None:
        tags = {"MSI band 04 bandpass adjustment slope and offset": "0.976500, 0.000900"}
        self.assertEqual(parse_hls_bandpass_tags(tags), {SpectralRole.RED: (0.9765, 0.0009)})


if __name__ == "__main__":
    unittest.main()
