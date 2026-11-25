"""Collects per-pair statistics as pairs are processed, so pixel arrays can be released straight away."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..comparison.comparison_run import VARIABLES, Condition, LandcoverLayers, PairComparison
from ..preprocessing.landcover import NO_CLASS
from .residual_analyzer import ResidualAccumulator, ResidualMoments, summarize_residuals

ALL_CLASSES = 0


@dataclass
class StatisticsCollector:
    pair_rows: list[dict[str, Any]] = field(default_factory=list)
    pooled: dict[tuple[Condition, str, int], ResidualAccumulator] = field(default_factory=lambda: defaultdict(ResidualAccumulator))
    moments: dict[tuple[Condition, str, int], dict[str, ResidualMoments]] = field(default_factory=lambda: defaultdict(dict))
    denominator_rows: list[dict[str, Any]] = field(default_factory=list)

    def add_pair(self, pair_id: str, comparison: PairComparison, landcover: LandcoverLayers) -> None:
        self._record_denominators(pair_id, comparison, landcover)
        for condition, arrays in comparison.conditions.items():
            classes = landcover.by_name(arrays.grid)
            present = [int(value) for value in np.unique(classes) if value != NO_CLASS]
            for variable in VARIABLES:
                landsat_values = arrays.landsat[variable]
                sentinel2_values = arrays.sentinel2[variable]
                self._record(pair_id, condition, variable, ALL_CLASSES, landsat_values, sentinel2_values)
                for land_class in present:
                    in_class = classes == land_class
                    self._record(
                        pair_id,
                        condition,
                        variable,
                        land_class,
                        np.where(in_class, landsat_values, np.nan),
                        np.where(in_class, sentinel2_values, np.nan),
                    )

    def _record_denominators(self, pair_id: str, comparison: PairComparison, landcover: LandcoverLayers) -> None:
        """NDVI's denominator (red + NIR) per land-cover class on the common grid: small values make NDVI unstable."""
        arrays = comparison.conditions.get(Condition.C1_AREA_WEIGHTED)
        if arrays is None:
            return
        for label, values in (("landsat", arrays.landsat), ("sentinel2", arrays.sentinel2)):
            denominator = values["red"] + values["nir"]
            for land_class in [ALL_CLASSES, *(int(v) for v in np.unique(landcover.landsat) if v != NO_CLASS)]:
                selected = denominator if land_class == ALL_CLASSES else np.where(landcover.landsat == land_class, denominator, np.nan)
                finite = selected[np.isfinite(selected)]
                if finite.size:
                    self.denominator_rows.append(
                        {"pair_id": pair_id, "sensor": label, "landcover_class": land_class, "count": int(finite.size),
                         "median": float(np.median(finite)), "share_below_0_1": float((finite < 0.1).mean())}
                    )

    def _record(self, pair_id: str, condition: Condition, variable: str, land_class: int, landsat: np.ndarray, sentinel2: np.ndarray) -> None:
        summary = summarize_residuals(landsat, sentinel2)
        if summary.count == 0:
            return
        self.pair_rows.append(
            {"pair_id": pair_id, "condition": condition.value, "variable": variable, "landcover_class": land_class, **summary.as_dict()}
        )
        key = (condition, variable, land_class)
        self.pooled[key].add(landsat, sentinel2)
        self.moments[key][pair_id] = ResidualMoments.from_arrays(
            landsat[np.isfinite(landsat) & np.isfinite(sentinel2)], sentinel2[np.isfinite(landsat) & np.isfinite(sentinel2)]
        )

    def pair_ids_with(self, conditions: tuple[Condition, ...], variable: str = "ndvi") -> list[str]:
        sets = [set(self.moments.get((condition, variable, ALL_CLASSES), {})) for condition in conditions]
        return sorted(set.intersection(*sets)) if sets else []

    def moments_for(self, condition: Condition, variable: str, pair_ids: list[str], land_class: int = ALL_CLASSES) -> list[ResidualMoments]:
        table = self.moments.get((condition, variable, land_class), {})
        return [table.get(pair_id, ResidualMoments()) for pair_id in pair_ids]
