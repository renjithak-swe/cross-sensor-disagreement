"""Digital numbers to surface reflectance, per product."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import MetadataError

# USGS Landsat 8-9 OLI/TIRS Collection 2 Level-2 Science Product Guide: valid surface reflectance DN range.
LANDSAT_SR_VALID_DN = (7273, 43636)
SENTINEL2_NODATA_DN = 0
SENTINEL2_SATURATED_DN = 65535
PHYSICAL_REFLECTANCE_RANGE = (0.0, 1.0)


@dataclass(frozen=True)
class LinearScaling:
    """reflectance = dn * scale + offset, for dn in [valid_min_dn, valid_max_dn] and dn != nodata_dn."""

    scale: float
    offset: float
    nodata_dn: int | None
    valid_min_dn: int | None
    valid_max_dn: int | None
    source: str


def landsat_scaling(raster_bands: tuple[dict[str, Any], ...], asset_key: str) -> LinearScaling:
    """Scale and offset come from the item's raster:bands; they are not assumed."""
    if len(raster_bands) != 1:
        raise MetadataError(f"Landsat asset {asset_key} must describe exactly one raster band")
    band = raster_bands[0]
    try:
        scale = float(band["scale"])
        offset = float(band["offset"])
    except (KeyError, TypeError, ValueError) as error:
        raise MetadataError(f"Landsat asset {asset_key} has no usable scale/offset in raster:bands") from error
    nodata = band.get("nodata")
    return LinearScaling(
        scale=scale,
        offset=offset,
        nodata_dn=None if nodata is None else int(nodata),
        valid_min_dn=LANDSAT_SR_VALID_DN[0],
        valid_max_dn=LANDSAT_SR_VALID_DN[1],
        source=f"STAC raster:bands of {asset_key}",
    )


def sentinel2_scaling(quantification_value: float, add_offset: float, band: str) -> LinearScaling:
    """Sentinel-2 L2A: reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE."""
    if quantification_value <= 0:
        raise MetadataError("BOA_QUANTIFICATION_VALUE must be positive")
    return LinearScaling(
        scale=1.0 / quantification_value,
        offset=add_offset / quantification_value,
        nodata_dn=SENTINEL2_NODATA_DN,
        valid_min_dn=None,
        valid_max_dn=SENTINEL2_SATURATED_DN - 1,
        source=f"product metadata BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET for {band}",
    )


def file_scaling(scale: float | None, offset: float | None, nodata: float | None, label: str) -> LinearScaling:
    """Scaling declared in the GeoTIFF itself (used for HLS)."""
    if scale is None or scale <= 0:
        raise MetadataError(f"{label} declares no positive scale factor")
    return LinearScaling(
        scale=scale,
        offset=offset or 0.0,
        nodata_dn=None if nodata is None else int(nodata),
        valid_min_dn=None,
        valid_max_dn=None,
        source=f"GeoTIFF scale/offset of {label}",
    )


def apply_scaling(digital_numbers: np.ndarray, scaling: LinearScaling) -> np.ndarray:
    """Return float32 reflectance with NaN wherever the DN is nodata, outside its valid range, or non-physical."""
    dn = digital_numbers.astype(np.float64)
    is_invalid = np.zeros(dn.shape, dtype=bool)
    if scaling.nodata_dn is not None:
        is_invalid |= digital_numbers == scaling.nodata_dn
    if scaling.valid_min_dn is not None:
        is_invalid |= dn < scaling.valid_min_dn
    if scaling.valid_max_dn is not None:
        is_invalid |= dn > scaling.valid_max_dn
    reflectance = dn * scaling.scale + scaling.offset
    is_invalid |= ~((reflectance > PHYSICAL_REFLECTANCE_RANGE[0]) & (reflectance <= PHYSICAL_REFLECTANCE_RANGE[1]))
    reflectance[is_invalid] = np.nan
    return reflectance.astype(np.float32)
