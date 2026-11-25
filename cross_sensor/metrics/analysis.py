"""Turns collected statistics into the summary the report and the article draw on."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..comparison.comparison_run import CONDITION_LABELS, LADDER, SPATIAL_METHODS, Condition
from ..preprocessing.landcover import NO_CLASS, WORLDCOVER_CLASSES
from .statistics_collector import ALL_CLASSES, StatisticsCollector
from .uncertainty import METRICS, Interval, bootstrap_paired_difference, bootstrap_pooled, bootstrap_spearman

if TYPE_CHECKING:
    from ..pipeline import RunResult

SAME_DAY_HOURS = 1.0
MIN_CLASS_PAIRS = 3
MIN_CLASS_PIXELS = 1000

SIDE_COMPARISONS = (
    (Condition.C1_AREA_WEIGHTED, Condition.C2_BILINEAR, "GDAL bilinear instead of area-weighted aggregation"),
    (Condition.C1_AREA_WEIGHTED, Condition.C3_NEAREST, "GDAL nearest neighbour instead of area-weighted aggregation"),
    (Condition.C1_AREA_WEIGHTED, Condition.C4_HLS_GRID, "HLS-style grid (Landsat cubic-resampled) instead of the Landsat grid"),
    (Condition.C1_AREA_WEIGHTED, Condition.E0_VIEW_ANGLE_BROAD_NIR, "view-angle normalization alone, with broad NIR"),
    (Condition.D1_NARROW_NIR, Condition.D2_BANDPASS, "bandpass adjustment alone, without view-angle normalization"),
    (Condition.C4_HLS_CLEAR, Condition.F_HLS, "HLS products instead of the same-grid L2 comparison, on the same cells"),
)


def _interval_dict(interval: Interval) -> dict[str, float | int]:
    return interval.as_dict()


def _pooled_block(statistics: StatisticsCollector, condition: Condition, variable: str, pair_ids: list[str], iterations: int, rng: np.random.Generator, land_class: int = ALL_CLASSES) -> dict[str, Any]:
    moments = statistics.moments_for(condition, variable, pair_ids, land_class)
    block: dict[str, Any] = {"pixels": int(sum(m.count for m in moments)), "pairs": int(sum(1 for m in moments if m.count))}
    for name, metric in METRICS.items():
        block[name] = _interval_dict(bootstrap_pooled(moments, metric, iterations, rng))
    return block


def _difference_block(statistics: StatisticsCollector, first: Condition, second: Condition, variable: str, pair_ids: list[str], iterations: int, rng: np.random.Generator) -> dict[str, Any]:
    first_moments = statistics.moments_for(first, variable, pair_ids)
    second_moments = statistics.moments_for(second, variable, pair_ids)
    return {
        name: _interval_dict(bootstrap_paired_difference(first_moments, second_moments, metric, iterations, rng))
        for name, metric in METRICS.items()
    }


def _pair_level(statistics: StatisticsCollector, condition: Condition, pair_ids: list[str]) -> dict[str, float | int | None]:
    """Distribution of per-pair NDVI MAE, the alternative to the pixel-weighted pooled estimate."""
    wanted = set(pair_ids)
    values = np.array([row["mae"] for row in statistics.pair_rows if row["condition"] == condition.value and row["variable"] == "ndvi" and row["landcover_class"] == ALL_CLASSES and row["pair_id"] in wanted])
    if values.size == 0:
        return {"pairs": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {"pairs": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "q1": float(np.percentile(values, 25)), "q3": float(np.percentile(values, 75))}


def _denominator_summary(statistics: StatisticsCollector) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for land_class, name in [(ALL_CLASSES, "All cells"), *WORLDCOVER_CLASSES.items()]:
        rows = [row for row in statistics.denominator_rows if row["landcover_class"] == land_class]
        if not rows:
            continue
        summary[name] = {
            sensor: {
                "pairs": len(subset),
                "cells": sum(row["count"] for row in subset),
                "median_of_pair_medians": float(np.median([row["median"] for row in subset])),
                "share_below_0_1": sum(row["share_below_0_1"] * row["count"] for row in subset) / sum(row["count"] for row in subset),
            }
            for sensor in ("landsat", "sentinel2")
            if (subset := [row for row in rows if row["sensor"] == sensor])
        }
    return summary


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else value


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def build_summary(result: RunResult, iterations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    statistics = result.statistics

    observation_counts: dict[str, dict[str, int]] = {}
    for record in result.observations:
        observation_counts.setdefault(record.sensor, Counter())[record.decision.reason.value] += 1
    pair_counts = Counter(record.decision.reason.value for record in result.pairs)
    compared = [record for record in result.pairs if record.decision.is_accepted and record.valid_fraction is not None]
    compared_ids = sorted(record.pair.pair_id for record in compared)

    pooled: dict[str, Any] = {}
    for condition in Condition:
        available = statistics.pair_ids_with((condition,))
        if not available:
            continue
        pooled[condition.value] = {
            variable: {
                **_pooled_block(statistics, condition, variable, available, iterations, rng),
                "median_ae_pooled": statistics.pooled[(condition, variable, ALL_CLASSES)].summary().median_absolute_error,
                "residual_sd_pooled": statistics.pooled[(condition, variable, ALL_CLASSES)].summary().residual_standard_deviation,
                "pearson_r_pooled": statistics.pooled[(condition, variable, ALL_CLASSES)].summary().pearson_r,
            }
            for variable in ("ndvi", "red", "nir")
        }

    ladder_ids = statistics.pair_ids_with(LADDER)
    ladder = {
        "pairs": ladder_ids,
        "steps": [
            {"condition": condition.value, **{variable: _pooled_block(statistics, condition, variable, ladder_ids, iterations, rng) for variable in ("ndvi", "red", "nir")}}
            for condition in LADDER
        ],
        "pair_level_ndvi_mae": {condition.value: _pair_level(statistics, condition, ladder_ids) for condition in LADDER},
        "step_changes": [
            {"from": first.value, "to": second.value, **{variable: _difference_block(statistics, first, second, variable, ladder_ids, iterations, rng) for variable in ("ndvi", "red", "nir")}}
            for first, second in zip(LADDER, LADDER[1:])
        ],
    }

    spatial_ids = statistics.pair_ids_with(SPATIAL_METHODS)
    spatial = {
        "pairs": spatial_ids,
        "methods": [
            {"condition": condition.value, "ndvi": _pooled_block(statistics, condition, "ndvi", spatial_ids, iterations, rng)}
            for condition in SPATIAL_METHODS
        ],
    }

    side: list[dict[str, Any]] = []
    for first, second, description in SIDE_COMPARISONS:
        pair_ids = statistics.pair_ids_with((first, second))
        if not pair_ids:
            continue
        side.append(
            {
                "from": first.value,
                "to": second.value,
                "description": description,
                "pairs": len(pair_ids),
                "ndvi_from": _pooled_block(statistics, first, "ndvi", pair_ids, iterations, rng),
                "ndvi_to": _pooled_block(statistics, second, "ndvi", pair_ids, iterations, rng),
                "ndvi_change": _difference_block(statistics, first, second, "ndvi", pair_ids, iterations, rng),
            }
        )

    landcover_breakdown: dict[str, Any] = {}
    for condition in (Condition.C1_AREA_WEIGHTED, Condition.D1_NARROW_NIR, Condition.E2_VIEW_ANGLE_BANDPASS):
        pair_ids = ladder_ids if condition in LADDER and ladder_ids else statistics.pair_ids_with((condition,))
        classes: dict[str, Any] = {}
        for land_class, name in WORLDCOVER_CLASSES.items():
            block = _pooled_block(statistics, condition, "ndvi", pair_ids, iterations, rng, land_class)
            if block["pairs"] >= MIN_CLASS_PAIRS and block["pixels"] >= MIN_CLASS_PIXELS:
                classes[name] = block
        landcover_breakdown[condition.value] = {"pairs": len(pair_ids), "classes": classes}

    per_pair = {row["pair_id"]: row for row in statistics.pair_rows if row["landcover_class"] == ALL_CLASSES and row["variable"] == "ndvi" and row["condition"] == Condition.C1_AREA_WEIGHTED.value}
    per_pair_d1 = {row["pair_id"]: row for row in statistics.pair_rows if row["landcover_class"] == ALL_CLASSES and row["variable"] == "ndvi" and row["condition"] == Condition.D1_NARROW_NIR.value}
    per_pair_e1 = {row["pair_id"]: row for row in statistics.pair_rows if row["landcover_class"] == ALL_CLASSES and row["variable"] == "ndvi" and row["condition"] == Condition.E1_VIEW_ANGLE.value}

    temporal_rows = [
        {
            "pair_id": record.pair.pair_id,
            "hours": abs(record.observed_signed_hours),
            "signed_hours": record.observed_signed_hours,
            "c1_ndvi_mae": per_pair[record.pair.pair_id]["mae"],
            "c1_ndvi_bias": per_pair[record.pair.pair_id]["mean_bias"],
        }
        for record in compared
        if record.pair.pair_id in per_pair
    ]
    hours = np.array([row["hours"] for row in temporal_rows], dtype=float)
    maes = np.array([row["c1_ndvi_mae"] for row in temporal_rows], dtype=float)
    same_day_ids = sorted(row["pair_id"] for row in temporal_rows if row["hours"] < SAME_DAY_HOURS)
    later_ids = sorted(row["pair_id"] for row in temporal_rows if row["hours"] >= SAME_DAY_HOURS)
    temporal = {
        "per_pair": temporal_rows,
        "spearman_hours_vs_c1_ndvi_mae": _interval_dict(bootstrap_spearman(hours, maes, iterations, rng)),
        "same_day": {"pair_ids": same_day_ids, "ndvi": _pooled_block(statistics, Condition.C1_AREA_WEIGHTED, "ndvi", same_day_ids, iterations, rng), "pair_level_mae": _pair_level(statistics, Condition.C1_AREA_WEIGHTED, same_day_ids)},
        "one_to_three_days": {"pair_ids": later_ids, "ndvi": _pooled_block(statistics, Condition.C1_AREA_WEIGHTED, "ndvi", later_ids, iterations, rng), "pair_level_mae": _pair_level(statistics, Condition.C1_AREA_WEIGHTED, later_ids)},
    }

    geometry_rows = []
    for record in compared:
        angles = record.angle_summary
        pair_id = record.pair.pair_id
        if not angles or pair_id not in per_pair_d1 or pair_id not in per_pair_e1:
            continue
        geometry_rows.append(
            {
                "pair_id": pair_id,
                "view_azimuth_difference_deg": _angle_difference(angles["s2_view_azimuth_mean"], angles["landsat_view_azimuth_mean"]),
                "s2_view_zenith_deg": angles["s2_view_zenith_mean"],
                "landsat_view_zenith_deg": angles["landsat_view_zenith_mean"],
                "sun_zenith_difference_deg": angles["s2_sun_zenith_mean"] - angles["landsat_sun_zenith_mean"],
                "d1_ndvi_mae": per_pair_d1[pair_id]["mae"],
                "e1_ndvi_mae": per_pair_e1[pair_id]["mae"],
                "e1_minus_d1_ndvi_mae": per_pair_e1[pair_id]["mae"] - per_pair_d1[pair_id]["mae"],
            }
        )
    azimuth_difference = np.array([row["view_azimuth_difference_deg"] for row in geometry_rows], dtype=float)
    geometry = {
        "per_pair": geometry_rows,
        "spearman_view_azimuth_difference_vs_d1_ndvi_mae": _interval_dict(
            bootstrap_spearman(azimuth_difference, np.array([row["d1_ndvi_mae"] for row in geometry_rows], dtype=float), iterations, rng)
        ),
        "spearman_view_azimuth_difference_vs_view_angle_gain": _interval_dict(
            bootstrap_spearman(azimuth_difference, np.array([row["e1_minus_d1_ndvi_mae"] for row in geometry_rows], dtype=float), iterations, rng)
        ),
    }

    aerosol_rows = [
        {
            "pair_id": record.pair.pair_id,
            "s30_high_aerosol_fraction": record.angle_summary["s30_high_aerosol_fraction"],
            "l30_high_aerosol_fraction": record.angle_summary["l30_high_aerosol_fraction"],
            "hls_clear_fraction_of_c4_cells": record.angle_summary["hls_clear_fraction_of_c4_cells"],
        }
        for record in compared
        if "s30_high_aerosol_fraction" in record.angle_summary
    ]
    hls_aerosol = {
        "per_pair": aerosol_rows,
        "median_s30_high_aerosol_fraction": float(np.median([row["s30_high_aerosol_fraction"] for row in aerosol_rows])) if aerosol_rows else None,
        "median_l30_high_aerosol_fraction": float(np.median([row["l30_high_aerosol_fraction"] for row in aerosol_rows])) if aerosol_rows else None,
        "pairs_with_majority_high_aerosol_in_either": sum(
            1 for row in aerosol_rows if max(row["s30_high_aerosol_fraction"], row["l30_high_aerosol_fraction"]) > 0.5
        ),
    }

    bandpass_checks = [
        {"pair_id": record.pair.pair_id, "platform": record.pair.sentinel2.platform if record.pair.sentinel2 else None, "hls_tags": record.provenance.hls_bandpass_tags}
        for record in compared
        if record.provenance.hls_bandpass_tags
    ]

    pure_counts = {
        WORLDCOVER_CLASSES.get(int(value), str(int(value))): int(count)
        for value, count in zip(*np.unique(result.landcover.landsat[result.grids.inside_aoi_landsat], return_counts=True), strict=True)
        if value != NO_CLASS
    }
    inside = int(result.grids.inside_aoi_landsat.sum())
    return {
        "aoi": {"name": result.aoi.name, "sha256": result.aoi.sha256, "landsat_grid_cells": inside, "area_km2": inside * result.grids.landsat.resolution**2 / 1e6},
        "counts": {"observations": {sensor: dict(counts) for sensor, counts in observation_counts.items()}, "pairs": dict(pair_counts), "compared_pairs": len(compared_ids)},
        "compared_pair_ids": compared_ids,
        "conditions": {condition.value: CONDITION_LABELS[condition] for condition in Condition},
        "landcover": {
            "source": result.landcover_source,
            "area_fraction": {WORLDCOVER_CLASSES.get(value, str(value)): fraction for value, fraction in sorted(result.landcover_fractions.items())},
            "pure_cells_on_landsat_grid": pure_counts,
            "impure_cells_on_landsat_grid": inside - sum(pure_counts.values()),
        },
        "pooled": pooled,
        "ladder": ladder,
        "spatial_methods": spatial,
        "side_comparisons": side,
        "landcover_breakdown": landcover_breakdown,
        "temporal": temporal,
        "geometry": geometry,
        "hls_bandpass_tag_check": bandpass_checks,
        "hls_aerosol_flags": hls_aerosol,
        "ndvi_denominator_by_class": _denominator_summary(statistics),
        "estimands": "pooled = every valid cell of every pair weighted equally (pixel-weighted); pair_level = the distribution of the 47 per-pair values",
        "bootstrap": {"iterations": iterations, "seed": seed, "unit": "observation pair", "confidence": 0.95},
    }
