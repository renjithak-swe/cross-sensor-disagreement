"""ESA WorldCover classes on the comparison grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..harmonization.grid_alignment import GridDefinition, Reducer, block_reduce

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
NO_CLASS = 0


@dataclass(frozen=True)
class DominantClass:
    classes: np.ndarray
    purity: np.ndarray

    def pure_classes(self, min_purity: float) -> np.ndarray:
        """Dominant class where it covers at least ``min_purity`` of the cell, NO_CLASS elsewhere."""
        return np.where(self.purity >= min_purity, self.classes, NO_CLASS).astype(np.int16)


def dominant_class(fine_classes: np.ndarray, fine_grid: GridDefinition, coarse_grid: GridDefinition) -> DominantClass:
    best_class = np.zeros(coarse_grid.shape, dtype=np.int16)
    best_fraction = np.zeros(coarse_grid.shape, dtype=np.float32)
    for value in np.unique(fine_classes):
        if value == NO_CLASS:
            continue
        fraction = block_reduce((fine_classes == value).astype(np.float32), fine_grid, coarse_grid, Reducer.MEAN)
        is_better = fraction > best_fraction
        best_class[is_better] = value
        best_fraction[is_better] = fraction[is_better]
    return DominantClass(classes=best_class, purity=best_fraction)


def class_area_fractions(fine_classes: np.ndarray, inside_aoi: np.ndarray) -> dict[int, float]:
    values = fine_classes[inside_aoi]
    if values.size == 0:
        return {}
    classes, counts = np.unique(values, return_counts=True)
    return {int(value): float(count) / values.size for value, count in zip(classes, counts, strict=True)}
