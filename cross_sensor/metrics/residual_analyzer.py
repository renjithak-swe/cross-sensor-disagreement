"""Residual statistics between paired observations. Residual = Sentinel-2 minus Landsat.

Correlation is reported, but never alone: two series can correlate almost perfectly and still differ by a
constant bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HISTOGRAM_RANGE = (-1.0, 1.0)
HISTOGRAM_BINS = 10_000
HISTOGRAM_EDGES = np.linspace(*HISTOGRAM_RANGE, HISTOGRAM_BINS + 1)
BIN_WIDTH = (HISTOGRAM_RANGE[1] - HISTOGRAM_RANGE[0]) / HISTOGRAM_BINS


@dataclass(frozen=True)
class ResidualSummary:
    count: int
    mean_bias: float
    mean_absolute_error: float
    median_absolute_error: float
    root_mean_square_error: float
    residual_standard_deviation: float
    pearson_r: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_bias": self.mean_bias,
            "mae": self.mean_absolute_error,
            "median_ae": self.median_absolute_error,
            "rmse": self.root_mean_square_error,
            "residual_sd": self.residual_standard_deviation,
            "pearson_r": self.pearson_r,
        }


EMPTY_SUMMARY = ResidualSummary(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)


def _pearson(count: int, sum_x: float, sum_y: float, sum_xx: float, sum_yy: float, sum_xy: float) -> float:
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_xx - sum_x**2 / count
    variance_y = sum_yy - sum_y**2 / count
    if variance_x <= 0 or variance_y <= 0:
        return math.nan
    return covariance / math.sqrt(variance_x * variance_y)


def summarize_residuals(landsat: np.ndarray, sentinel2: np.ndarray) -> ResidualSummary:
    """Exact statistics for one set of paired values. NaN in either input drops the pair."""
    if landsat.shape != sentinel2.shape:
        raise ValueError("paired arrays differ in shape")
    is_paired = np.isfinite(landsat) & np.isfinite(sentinel2)
    x = landsat[is_paired].astype(np.float64)
    y = sentinel2[is_paired].astype(np.float64)
    if x.size == 0:
        return EMPTY_SUMMARY
    residual = y - x
    bias = float(residual.mean())
    return ResidualSummary(
        count=int(x.size),
        mean_bias=bias,
        mean_absolute_error=float(np.abs(residual).mean()),
        median_absolute_error=float(np.median(np.abs(residual))),
        root_mean_square_error=float(np.sqrt(np.mean(residual**2))),
        residual_standard_deviation=float(residual.std()),
        pearson_r=_pearson(x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()) if x.size > 1 else math.nan,
    )


@dataclass
class ResidualMoments:
    """Additive sufficient statistics, so pairs can be pooled and bootstrapped without keeping pixels."""

    count: int = 0
    sum_residual: float = 0.0
    sum_absolute: float = 0.0
    sum_squared: float = 0.0

    @classmethod
    def from_arrays(cls, landsat: np.ndarray, sentinel2: np.ndarray) -> ResidualMoments:
        residual = (sentinel2.astype(np.float64) - landsat.astype(np.float64)).ravel()
        residual = residual[np.isfinite(residual)]
        return cls(int(residual.size), float(residual.sum()), float(np.abs(residual).sum()), float((residual**2).sum()))


@dataclass
class ResidualAccumulator:
    """Pooled statistics over many pairs. The median comes from a fixed histogram, accurate to BIN_WIDTH."""

    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0
    sum_residual: float = 0.0
    sum_absolute: float = 0.0
    sum_squared: float = 0.0
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(HISTOGRAM_BINS, dtype=np.int64))

    def add(self, landsat: np.ndarray, sentinel2: np.ndarray) -> None:
        is_paired = np.isfinite(landsat) & np.isfinite(sentinel2)
        x = landsat[is_paired].astype(np.float64)
        y = sentinel2[is_paired].astype(np.float64)
        if x.size == 0:
            return
        residual = y - x
        self.count += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float((x * x).sum())
        self.sum_yy += float((y * y).sum())
        self.sum_xy += float((x * y).sum())
        self.sum_residual += float(residual.sum())
        self.sum_absolute += float(np.abs(residual).sum())
        self.sum_squared += float((residual**2).sum())
        clipped = np.clip(residual, HISTOGRAM_RANGE[0], HISTOGRAM_RANGE[1] - BIN_WIDTH / 2)
        self.histogram += np.histogram(clipped, bins=HISTOGRAM_EDGES)[0]

    def absolute_histogram(self) -> np.ndarray:
        half = HISTOGRAM_BINS // 2
        return self.histogram[half:] + self.histogram[:half][::-1]

    def summary(self) -> ResidualSummary:
        if self.count == 0:
            return EMPTY_SUMMARY
        bias = self.sum_residual / self.count
        absolute = self.absolute_histogram()
        median_bin = int(np.searchsorted(np.cumsum(absolute), self.count / 2))
        return ResidualSummary(
            count=self.count,
            mean_bias=bias,
            mean_absolute_error=self.sum_absolute / self.count,
            median_absolute_error=(median_bin + 0.5) * BIN_WIDTH,
            root_mean_square_error=math.sqrt(self.sum_squared / self.count),
            residual_standard_deviation=math.sqrt(max(self.sum_squared / self.count - bias**2, 0.0)),
            pearson_r=_pearson(self.count, self.sum_x, self.sum_y, self.sum_xx, self.sum_yy, self.sum_xy),
        )
