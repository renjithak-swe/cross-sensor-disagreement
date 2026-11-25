"""Bootstrap intervals that resample whole observation pairs.

Pixels inside one pair share an atmosphere, a date and a geometry, so they are not independent. Resampling
pixels would give intervals that are far too narrow; the pair is the unit of resampling here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .residual_analyzer import ResidualMoments

CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    pair_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {"estimate": self.estimate, "ci_lower": self.lower, "ci_upper": self.upper, "pairs": self.pair_count}


MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def mean_absolute_error(count: np.ndarray, total: np.ndarray, absolute: np.ndarray, squared: np.ndarray) -> np.ndarray:
    return absolute / count


def mean_bias(count: np.ndarray, total: np.ndarray, absolute: np.ndarray, squared: np.ndarray) -> np.ndarray:
    return total / count


def root_mean_square_error(count: np.ndarray, total: np.ndarray, absolute: np.ndarray, squared: np.ndarray) -> np.ndarray:
    return np.sqrt(squared / count)


METRICS: dict[str, MetricFunction] = {"mae": mean_absolute_error, "mean_bias": mean_bias, "rmse": root_mean_square_error}


def _stack(moments: Sequence[ResidualMoments]) -> np.ndarray:
    return np.array([[m.count, m.sum_residual, m.sum_absolute, m.sum_squared] for m in moments], dtype=np.float64)


def _evaluate(metric: MetricFunction, sums: np.ndarray) -> np.ndarray:
    return metric(sums[..., 0], sums[..., 1], sums[..., 2], sums[..., 3])


def _bounds(samples: np.ndarray) -> tuple[float, float]:
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def bootstrap_pooled(moments: Sequence[ResidualMoments], metric: MetricFunction, iterations: int, rng: np.random.Generator) -> Interval:
    """Pixel-weighted pooled metric with a percentile interval from resampling pairs."""
    usable = [m for m in moments if m.count > 0]
    if not usable:
        return Interval(math.nan, math.nan, math.nan, 0)
    table = _stack(usable)
    estimate = float(_evaluate(metric, table.sum(axis=0)))
    if len(usable) < 2:
        return Interval(estimate, math.nan, math.nan, len(usable))
    indices = rng.integers(0, len(usable), size=(iterations, len(usable)))
    lower, upper = _bounds(_evaluate(metric, table[indices].sum(axis=1)))
    return Interval(estimate, lower, upper, len(usable))


def bootstrap_paired_difference(
    first: Sequence[ResidualMoments],
    second: Sequence[ResidualMoments],
    metric: MetricFunction,
    iterations: int,
    rng: np.random.Generator,
) -> Interval:
    """Metric(second) minus metric(first) on the same pairs, resampling pairs jointly."""
    if len(first) != len(second):
        raise ValueError("paired difference needs the same pairs in both conditions")
    kept = [(a, b) for a, b in zip(first, second, strict=True) if a.count > 0 and b.count > 0]
    if not kept:
        return Interval(math.nan, math.nan, math.nan, 0)
    table_first = _stack([a for a, _ in kept])
    table_second = _stack([b for _, b in kept])
    estimate = float(_evaluate(metric, table_second.sum(axis=0)) - _evaluate(metric, table_first.sum(axis=0)))
    if len(kept) < 2:
        return Interval(estimate, math.nan, math.nan, len(kept))
    indices = rng.integers(0, len(kept), size=(iterations, len(kept)))
    samples = _evaluate(metric, table_second[indices].sum(axis=1)) - _evaluate(metric, table_first[indices].sum(axis=1))
    lower, upper = _bounds(samples)
    return Interval(estimate, lower, upper, len(kept))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, iterations: int, rng: np.random.Generator) -> Interval:
    """Spearman rank correlation between per-pair values, with a pair-resampling interval."""
    is_finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[is_finite], y[is_finite]
    if x.size < 3:
        return Interval(math.nan, math.nan, math.nan, int(x.size))

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        rank_a = _average_ranks(a)
        rank_b = _average_ranks(b)
        if rank_a.std() == 0 or rank_b.std() == 0:
            return math.nan
        return float(np.corrcoef(rank_a, rank_b)[0, 1])

    estimate = spearman(x, y)
    samples = np.array([spearman(x[idx], y[idx]) for idx in rng.integers(0, x.size, size=(iterations, x.size))])
    samples = samples[np.isfinite(samples)]
    lower, upper = _bounds(samples) if samples.size else (math.nan, math.nan)
    return Interval(estimate, lower, upper, int(x.size))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks
