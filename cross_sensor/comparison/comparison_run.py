"""One observation pair, compared under every documented condition on one shared set of valid pixels.

Conditions form a ladder (A -> C1 -> D1 -> E1 -> E2), each adding one harmonization step, plus side branches
that swap a single choice (resampling method, NIR band, step order) and the HLS products themselves (F).
All conditions of a pair use the same ground area, so a change in disagreement between two conditions comes
from the processing step and not from a different set of pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from affine import Affine

from ..harmonization.geometry_normalization import c_factor
from ..harmonization.grid_alignment import (
    GdalResampling,
    GridDefinition,
    Reducer,
    block_reduce,
    block_reduce_angle,
    coarsen_onto,
    replicate_to_grid,
    resample_with_gdal,
)
from ..harmonization.spectral_adjustment import adjust_to_oli
from ..preprocessing.band_mapping import SpectralRole
from .ndvi_comparison import compute_ndvi


class Condition(str, Enum):
    A_NATIVE = "A_native"
    C1_AREA_WEIGHTED = "C1_area_weighted"
    C2_BILINEAR = "C2_bilinear"
    C3_NEAREST = "C3_nearest"
    C4_HLS_GRID = "C4_hls_grid_cubic"
    C4_HLS_CLEAR = "C4_hls_grid_cubic_hls_clear_cells"
    D0_BROAD_NIR_20M = "D0_broad_nir_20m_support"
    D1_NARROW_NIR = "D1_narrow_nir"
    D2_BANDPASS = "D2_narrow_nir_bandpass"
    E0_VIEW_ANGLE_BROAD_NIR = "E0_broad_nir_view_angle"
    E1_VIEW_ANGLE = "E1_narrow_nir_view_angle"
    E2_VIEW_ANGLE_BANDPASS = "E2_narrow_nir_view_angle_bandpass"
    F_HLS = "F_hls_products"


CONDITION_LABELS = {
    Condition.A_NATIVE: "Native pixels (S2 10 m vs Landsat 30 m, overlap-weighted), S2 B08",
    Condition.C1_AREA_WEIGHTED: "Common 30 m Landsat grid, S2 area-weighted mean, S2 B08",
    Condition.C2_BILINEAR: "Common 30 m Landsat grid, S2 GDAL bilinear, S2 B08",
    Condition.C3_NEAREST: "Common 30 m Landsat grid, S2 GDAL nearest neighbour, S2 B08",
    Condition.C4_HLS_GRID: "30 m MGRS (HLS) grid, S2 3x3 mean, Landsat GDAL cubic, S2 B08",
    Condition.C4_HLS_CLEAR: "C4 restricted to cells HLS Fmask also calls clear (the cells F uses)",
    Condition.D0_BROAD_NIR_20M: "C1 with S2 B08 first averaged onto B8A's 20 m pixels: B8A's spatial support, B08's spectral response",
    Condition.D1_NARROW_NIR: "C1 with S2 narrow NIR B8A (20 m)",
    Condition.D2_BANDPASS: "D1 + HLS v2.0 bandpass adjustment of S2",
    Condition.E0_VIEW_ANGLE_BROAD_NIR: "C1 + c-factor view-angle normalization, S2 B08",
    Condition.E1_VIEW_ANGLE: "D1 + c-factor view-angle normalization",
    Condition.E2_VIEW_ANGLE_BANDPASS: "E1 + HLS v2.0 bandpass adjustment of S2",
    Condition.F_HLS: "HLS v2.0 L30 vs S30 products (MGRS grid)",
}
LADDER = (
    Condition.A_NATIVE,
    Condition.C1_AREA_WEIGHTED,
    Condition.D0_BROAD_NIR_20M,
    Condition.D1_NARROW_NIR,
    Condition.E1_VIEW_ANGLE,
    Condition.E2_VIEW_ANGLE_BANDPASS,
)
SPATIAL_METHODS = (Condition.A_NATIVE, Condition.C1_AREA_WEIGHTED, Condition.C2_BILINEAR, Condition.C3_NEAREST, Condition.C4_HLS_GRID)
VARIABLES = ("red", "nir", "ndvi")


class GridName(str, Enum):
    FINE = "fine"
    LANDSAT = "landsat"
    MGRS = "mgrs"


@dataclass(frozen=True)
class ComparisonGrids:
    landsat: GridDefinition
    mgrs: GridDefinition
    fine: GridDefinition
    inside_aoi_landsat: np.ndarray
    read_bounds: tuple[float, float, float, float]

    def by_name(self, name: GridName) -> GridDefinition:
        return {GridName.FINE: self.fine, GridName.LANDSAT: self.landsat, GridName.MGRS: self.mgrs}[name]


@dataclass(frozen=True)
class LandcoverLayers:
    fine: np.ndarray
    landsat: np.ndarray
    mgrs: np.ndarray

    def by_name(self, name: GridName) -> np.ndarray:
        return {GridName.FINE: self.fine, GridName.LANDSAT: self.landsat, GridName.MGRS: self.mgrs}[name]


@dataclass(frozen=True)
class Layer:
    """A band in its native pixel grid. ``values`` is NaN wherever the pixel is unusable."""

    values: np.ndarray
    transform: Affine

    @property
    def grid(self) -> GridDefinition:
        return GridDefinition("", self.transform, self.values.shape[1], self.values.shape[0])


@dataclass(frozen=True)
class AngleLayers:
    """Per-pixel sun and view angles (degrees) in their native 30 m HLS windows; NaN where undefined."""

    sun_zenith: Layer
    sun_azimuth: Layer
    view_zenith: Layer
    view_azimuth: Layer


@dataclass(frozen=True)
class Sentinel2Inputs:
    platform: str
    red: Layer
    nir_broad: Layer
    nir_narrow: Layer
    is_invalid: Layer


@dataclass(frozen=True)
class LandsatInputs:
    red: Layer
    nir: Layer
    is_invalid: Layer


@dataclass(frozen=True)
class HlsInputs:
    red: np.ndarray
    nir: np.ndarray
    is_invalid: np.ndarray
    is_high_aerosol: np.ndarray
    angles: AngleLayers


@dataclass(frozen=True)
class PairInputs:
    crs: str
    sentinel2: Sentinel2Inputs
    landsat: LandsatInputs
    hls_sentinel2: HlsInputs | None
    hls_landsat: HlsInputs | None


@dataclass
class ConditionArrays:
    grid: GridName
    landsat: dict[str, np.ndarray]
    sentinel2: dict[str, np.ndarray]


@dataclass
class PairComparison:
    valid_fraction: float
    valid_landsat_cells: int
    conditions: dict[Condition, ConditionArrays] = field(default_factory=dict)
    unavailable: dict[Condition, str] = field(default_factory=dict)
    angle_summary: dict[str, float] = field(default_factory=dict)
    common_mask_sha256: str = ""


def _bool_layer_on(layer: Layer, grid: GridDefinition) -> np.ndarray:
    return replicate_to_grid(layer.values.astype(bool), layer.transform, grid)


def _values_on(layer: Layer, grid: GridDefinition) -> np.ndarray:
    return replicate_to_grid(layer.values, layer.transform, grid)


def _masked(values: np.ndarray, is_valid: np.ndarray) -> np.ndarray:
    return np.where(is_valid, values, np.nan).astype(np.float32)


def _variables(red: np.ndarray, nir: np.ndarray, is_valid: np.ndarray) -> dict[str, np.ndarray]:
    red_masked = _masked(red, is_valid)
    nir_masked = _masked(nir, is_valid)
    return {"red": red_masked, "nir": nir_masked, "ndvi": compute_ndvi(red_masked, nir_masked)}


def broad_nir_at_narrow_support(inputs: Sentinel2Inputs) -> Layer:
    """B08 averaged onto B8A's own 20 m pixels, so B08 and B8A can be compared at identical spatial support."""
    narrow = inputs.nir_narrow
    return Layer(coarsen_onto(inputs.nir_broad.values, inputs.nir_broad.transform, narrow.transform, narrow.values.shape), narrow.transform)


def sentinel2_invalid_fine(inputs: Sentinel2Inputs, fine: GridDefinition) -> np.ndarray:
    """True on fine cells where any Sentinel-2 layer used by any condition is unusable."""
    invalid = _bool_layer_on(inputs.is_invalid, fine)
    for layer in (inputs.red, inputs.nir_broad, inputs.nir_narrow, broad_nir_at_narrow_support(inputs)):
        invalid |= replicate_to_grid(np.isnan(layer.values), layer.transform, fine)
    return invalid


def landsat_invalid_on(inputs: LandsatInputs, grid: GridDefinition) -> np.ndarray:
    invalid = _bool_layer_on(inputs.is_invalid, grid)
    for layer in (inputs.red, inputs.nir):
        invalid |= replicate_to_grid(np.isnan(layer.values), layer.transform, grid)
    return invalid


def _sentinel2_invalid_10m(inputs: Sentinel2Inputs) -> np.ndarray:
    """Invalid mask on the 10 m red/B08 pixel grid, used for the GDAL resampling branches."""
    ten_metre = inputs.red.grid
    invalid = replicate_to_grid(inputs.is_invalid.values.astype(bool), inputs.is_invalid.transform, ten_metre)
    return invalid | np.isnan(inputs.red.values) | np.isnan(inputs.nir_broad.values)


def _angles_on_landsat(angles: AngleLayers, grids: ComparisonGrids) -> dict[str, np.ndarray]:
    def zenith(layer: Layer) -> np.ndarray:
        fine = replicate_to_grid(layer.values, layer.transform, grids.fine)
        return block_reduce(fine, grids.fine, grids.landsat, Reducer.MEAN)

    def azimuth(layer: Layer) -> np.ndarray:
        fine = replicate_to_grid(layer.values, layer.transform, grids.fine)
        result = block_reduce_angle(np.nan_to_num(fine), grids.fine, grids.landsat)
        is_undefined = block_reduce(np.isnan(fine).astype(np.float32), grids.fine, grids.landsat, Reducer.MEAN) > 0
        return np.where(is_undefined, np.nan, result).astype(np.float32)

    return {
        "sun_zenith": zenith(angles.sun_zenith),
        "sun_azimuth": azimuth(angles.sun_azimuth),
        "view_zenith": zenith(angles.view_zenith),
        "view_azimuth": azimuth(angles.view_azimuth),
    }


def _normalize_view_angle(
    red: np.ndarray, nir: np.ndarray, nir_role: SpectralRole, angles: dict[str, np.ndarray], target_sun_zenith: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    def factor(role: SpectralRole) -> np.ndarray:
        return c_factor(role, angles["sun_zenith"], angles["view_zenith"], angles["sun_azimuth"], angles["view_azimuth"], target_sun_zenith)

    return red * factor(SpectralRole.RED), nir * factor(nir_role)


def _circular_mean_degrees(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    radians = np.deg2rad(finite.astype(np.float64))
    return float(np.mod(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())), 360.0))


def compare_pair(inputs: PairInputs, grids: ComparisonGrids) -> PairComparison:
    fine, landsat_grid, mgrs = grids.fine, grids.landsat, grids.mgrs

    s2_invalid_fine = sentinel2_invalid_fine(inputs.sentinel2, fine)
    s2_valid_landsat = ~block_reduce(s2_invalid_fine, fine, landsat_grid, Reducer.ANY)
    landsat_invalid = landsat_invalid_on(inputs.landsat, landsat_grid)

    s2_invalid_10m = _sentinel2_invalid_10m(inputs.sentinel2)
    resampled: dict[Condition, dict[str, np.ndarray]] = {}
    resampled_valid = np.ones(landsat_grid.shape, dtype=bool)
    for condition, method in ((Condition.C2_BILINEAR, GdalResampling.BILINEAR), (Condition.C3_NEAREST, GdalResampling.NEAREST)):
        red, red_invalid = resample_with_gdal(inputs.sentinel2.red.values, inputs.sentinel2.red.transform, inputs.crs, landsat_grid, method, s2_invalid_10m)
        nir, _ = resample_with_gdal(inputs.sentinel2.nir_broad.values, inputs.sentinel2.nir_broad.transform, inputs.crs, landsat_grid, method, s2_invalid_10m)
        resampled[condition] = {"red": red, "nir": nir}
        resampled_valid &= ~red_invalid

    has_angles = inputs.hls_sentinel2 is not None and inputs.hls_landsat is not None
    angles_s2: dict[str, np.ndarray] = {}
    angles_landsat: dict[str, np.ndarray] = {}
    angles_valid = np.ones(landsat_grid.shape, dtype=bool)
    if has_angles:
        angles_s2 = _angles_on_landsat(inputs.hls_sentinel2.angles, grids)
        angles_landsat = _angles_on_landsat(inputs.hls_landsat.angles, grids)
        for layer in (*angles_s2.values(), *angles_landsat.values()):
            angles_valid &= np.isfinite(layer)

    common = grids.inside_aoi_landsat & s2_valid_landsat & ~landsat_invalid & resampled_valid & angles_valid
    aoi_cells = int(grids.inside_aoi_landsat.sum())
    comparison = PairComparison(
        valid_fraction=float(common.sum()) / aoi_cells if aoi_cells else 0.0,
        valid_landsat_cells=int(common.sum()),
        common_mask_sha256=hashlib.sha256(np.packbits(common).tobytes()).hexdigest(),
    )
    if not common.any():
        return comparison

    s2_red_fine = _values_on(inputs.sentinel2.red, fine)
    s2_nir_broad_fine = _values_on(inputs.sentinel2.nir_broad, fine)
    s2_nir_narrow_fine = _values_on(inputs.sentinel2.nir_narrow, fine)
    landsat_red = _values_on(inputs.landsat.red, landsat_grid)
    landsat_nir = _values_on(inputs.landsat.nir, landsat_grid)

    common_fine = replicate_to_grid(common, landsat_grid.transform, fine)
    comparison.conditions[Condition.A_NATIVE] = ConditionArrays(
        GridName.FINE,
        landsat=_variables(replicate_to_grid(landsat_red, landsat_grid.transform, fine), replicate_to_grid(landsat_nir, landsat_grid.transform, fine), common_fine),
        sentinel2=_variables(s2_red_fine, s2_nir_broad_fine, common_fine),
    )
    del common_fine

    s2_red = block_reduce(np.nan_to_num(s2_red_fine), fine, landsat_grid, Reducer.MEAN)
    s2_nir_broad = block_reduce(np.nan_to_num(s2_nir_broad_fine), fine, landsat_grid, Reducer.MEAN)
    s2_nir_narrow = block_reduce(np.nan_to_num(s2_nir_narrow_fine), fine, landsat_grid, Reducer.MEAN)
    s2_nir_broad_20m = block_reduce(np.nan_to_num(_values_on(broad_nir_at_narrow_support(inputs.sentinel2), fine)), fine, landsat_grid, Reducer.MEAN)
    landsat_variables = _variables(landsat_red, landsat_nir, common)
    comparison.conditions[Condition.C1_AREA_WEIGHTED] = ConditionArrays(GridName.LANDSAT, landsat_variables, _variables(s2_red, s2_nir_broad, common))
    for condition, bands in resampled.items():
        comparison.conditions[condition] = ConditionArrays(GridName.LANDSAT, landsat_variables, _variables(bands["red"], bands["nir"], common))
    comparison.conditions[Condition.D0_BROAD_NIR_20M] = ConditionArrays(GridName.LANDSAT, landsat_variables, _variables(s2_red, s2_nir_broad_20m, common))
    comparison.conditions[Condition.D1_NARROW_NIR] = ConditionArrays(GridName.LANDSAT, landsat_variables, _variables(s2_red, s2_nir_narrow, common))

    platform = inputs.sentinel2.platform
    bandpass_red = adjust_to_oli(s2_red, platform, SpectralRole.RED)
    bandpass_nir = adjust_to_oli(s2_nir_narrow, platform, SpectralRole.NIR_NARROW)
    comparison.conditions[Condition.D2_BANDPASS] = ConditionArrays(GridName.LANDSAT, landsat_variables, _variables(bandpass_red, bandpass_nir, common))

    _add_mgrs_conditions(comparison, inputs, grids, common, s2_red_fine, s2_nir_broad_fine)
    del s2_red_fine, s2_nir_broad_fine, s2_nir_narrow_fine

    if not has_angles:
        reason = "HLS angle layers unavailable for this pair"
        for condition in (Condition.E0_VIEW_ANGLE_BROAD_NIR, Condition.E1_VIEW_ANGLE, Condition.E2_VIEW_ANGLE_BANDPASS):
            comparison.unavailable[condition] = reason
        return comparison

    target_sun_zenith = (angles_s2["sun_zenith"] + angles_landsat["sun_zenith"]) / 2
    landsat_red_n, landsat_nir_n = _normalize_view_angle(landsat_red, landsat_nir, SpectralRole.NIR_NARROW, angles_landsat, target_sun_zenith)
    landsat_normalized = _variables(landsat_red_n, landsat_nir_n, common)
    s2_red_n, s2_broad_n = _normalize_view_angle(s2_red, s2_nir_broad, SpectralRole.NIR_BROAD, angles_s2, target_sun_zenith)
    comparison.conditions[Condition.E0_VIEW_ANGLE_BROAD_NIR] = ConditionArrays(GridName.LANDSAT, landsat_normalized, _variables(s2_red_n, s2_broad_n, common))
    s2_red_n, s2_narrow_n = _normalize_view_angle(s2_red, s2_nir_narrow, SpectralRole.NIR_NARROW, angles_s2, target_sun_zenith)
    comparison.conditions[Condition.E1_VIEW_ANGLE] = ConditionArrays(GridName.LANDSAT, landsat_normalized, _variables(s2_red_n, s2_narrow_n, common))
    comparison.conditions[Condition.E2_VIEW_ANGLE_BANDPASS] = ConditionArrays(
        GridName.LANDSAT,
        landsat_normalized,
        _variables(adjust_to_oli(s2_red_n, platform, SpectralRole.RED), adjust_to_oli(s2_narrow_n, platform, SpectralRole.NIR_NARROW), common),
    )
    comparison.angle_summary |= {
        "s2_view_zenith_mean": float(np.nanmean(np.where(common, angles_s2["view_zenith"], np.nan))),
        "s2_view_azimuth_mean": _circular_mean_degrees(np.where(common, angles_s2["view_azimuth"], np.nan)),
        "s2_sun_zenith_mean": float(np.nanmean(np.where(common, angles_s2["sun_zenith"], np.nan))),
        "s2_sun_azimuth_mean": _circular_mean_degrees(np.where(common, angles_s2["sun_azimuth"], np.nan)),
        "landsat_view_zenith_mean": float(np.nanmean(np.where(common, angles_landsat["view_zenith"], np.nan))),
        "landsat_view_azimuth_mean": _circular_mean_degrees(np.where(common, angles_landsat["view_azimuth"], np.nan)),
        "landsat_sun_zenith_mean": float(np.nanmean(np.where(common, angles_landsat["sun_zenith"], np.nan))),
        "landsat_sun_azimuth_mean": _circular_mean_degrees(np.where(common, angles_landsat["sun_azimuth"], np.nan)),
    }
    return comparison


def _add_mgrs_conditions(
    comparison: PairComparison,
    inputs: PairInputs,
    grids: ComparisonGrids,
    common: np.ndarray,
    s2_red_fine: np.ndarray,
    s2_nir_broad_fine: np.ndarray,
) -> None:
    """C4 (an HLS-like grid built from the same L2 products) and F (the HLS products), on the same MGRS cells."""
    fine, mgrs = grids.fine, grids.mgrs
    landsat_invalid_native = inputs.landsat.is_invalid.values.astype(bool) | np.isnan(inputs.landsat.red.values) | np.isnan(inputs.landsat.nir.values)
    landsat_red_cubic, cubic_invalid = resample_with_gdal(
        inputs.landsat.red.values, inputs.landsat.red.transform, inputs.crs, mgrs, GdalResampling.CUBIC, landsat_invalid_native
    )
    landsat_nir_cubic, _ = resample_with_gdal(
        inputs.landsat.nir.values, inputs.landsat.nir.transform, inputs.crs, mgrs, GdalResampling.CUBIC, landsat_invalid_native
    )
    common_mgrs = block_reduce(replicate_to_grid(common, grids.landsat.transform, fine), fine, mgrs, Reducer.ALL) & ~cubic_invalid

    s2_red_mgrs = block_reduce(np.nan_to_num(s2_red_fine), fine, mgrs, Reducer.MEAN)
    s2_nir_mgrs = block_reduce(np.nan_to_num(s2_nir_broad_fine), fine, mgrs, Reducer.MEAN)
    comparison.conditions[Condition.C4_HLS_GRID] = ConditionArrays(
        GridName.MGRS,
        landsat=_variables(landsat_red_cubic, landsat_nir_cubic, common_mgrs),
        sentinel2=_variables(s2_red_mgrs, s2_nir_mgrs, common_mgrs),
    )

    if inputs.hls_sentinel2 is None or inputs.hls_landsat is None:
        reason = "HLS L30/S30 granules unavailable for this pair"
        comparison.unavailable[Condition.F_HLS] = reason
        comparison.unavailable[Condition.C4_HLS_CLEAR] = reason
        return
    hls_clear = common_mgrs & ~inputs.hls_sentinel2.is_invalid & ~inputs.hls_landsat.is_invalid
    for hls in (inputs.hls_sentinel2, inputs.hls_landsat):
        hls_clear &= np.isfinite(hls.red) & np.isfinite(hls.nir)
    comparison.conditions[Condition.C4_HLS_CLEAR] = ConditionArrays(
        GridName.MGRS,
        landsat=_variables(landsat_red_cubic, landsat_nir_cubic, hls_clear),
        sentinel2=_variables(s2_red_mgrs, s2_nir_mgrs, hls_clear),
    )
    comparison.conditions[Condition.F_HLS] = ConditionArrays(
        GridName.MGRS,
        landsat=_variables(inputs.hls_landsat.red, inputs.hls_landsat.nir, hls_clear),
        sentinel2=_variables(inputs.hls_sentinel2.red, inputs.hls_sentinel2.nir, hls_clear),
    )
    comparison.angle_summary["hls_clear_fraction_of_c4_cells"] = float(hls_clear.sum()) / max(int(common_mgrs.sum()), 1)
    for label, hls in (("s30", inputs.hls_sentinel2), ("l30", inputs.hls_landsat)):
        comparison.angle_summary[f"{label}_high_aerosol_fraction"] = float(hls.is_high_aerosol[common_mgrs].mean()) if common_mgrs.any() else float("nan")
