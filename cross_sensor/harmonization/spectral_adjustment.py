"""Sentinel-2 MSI to Landsat OLI bandpass adjustment, using published HLS v2.0 coefficients only.

No coefficient is fitted to the comparison pairs: doing so and then reporting the agreement would be circular.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..errors import MetadataError, UnsupportedPlatformError
from ..preprocessing.band_mapping import SpectralRole

JU_2025_TABLE_5 = "Ju et al. (2025), RSE 324:114723, Table 5"
HLS_S30_PRODUCT_TAGS = "HLS v2.0 S30 GeoTIFF tags ('MSI band NN bandpass adjustment slope and offset')"


@dataclass(frozen=True)
class BandpassCoefficients:
    slope: float
    intercept: float
    source: str


HLS_V2_BANDPASS: dict[tuple[str, SpectralRole], BandpassCoefficients] = {
    ("sentinel-2a", SpectralRole.RED): BandpassCoefficients(0.9765, 0.0009, JU_2025_TABLE_5),
    ("sentinel-2a", SpectralRole.NIR_NARROW): BandpassCoefficients(0.9983, -0.0001, JU_2025_TABLE_5),
    ("sentinel-2b", SpectralRole.RED): BandpassCoefficients(0.9761, 0.0010, JU_2025_TABLE_5),
    ("sentinel-2b", SpectralRole.NIR_NARROW): BandpassCoefficients(0.9966, 0.0000, JU_2025_TABLE_5),
    ("sentinel-2c", SpectralRole.RED): BandpassCoefficients(0.9718, 0.0011, HLS_S30_PRODUCT_TAGS),
    ("sentinel-2c", SpectralRole.NIR_NARROW): BandpassCoefficients(0.9995, -0.0003, HLS_S30_PRODUCT_TAGS),
}

_HLS_TAG_BANDS = {SpectralRole.RED: "04", SpectralRole.NIR_NARROW: "8a"}


def coefficients_for(platform: str, role: SpectralRole) -> BandpassCoefficients:
    key = (platform.lower(), role)
    if key not in HLS_V2_BANDPASS:
        raise UnsupportedPlatformError(f"no published bandpass coefficients for {platform} {role.value}")
    return HLS_V2_BANDPASS[key]


def adjust_to_oli(reflectance: np.ndarray, platform: str, role: SpectralRole) -> np.ndarray:
    coefficients = coefficients_for(platform, role)
    return (reflectance * coefficients.slope + coefficients.intercept).astype(np.float32)


def parse_hls_bandpass_tags(tags: dict[str, str]) -> dict[SpectralRole, tuple[float, float]]:
    """Coefficients an HLS S30 granule says it applied, used to cross-check the table above."""
    parsed: dict[SpectralRole, tuple[float, float]] = {}
    for role, band in _HLS_TAG_BANDS.items():
        value = tags.get(f"MSI band {band} bandpass adjustment slope and offset")
        if value is None:
            continue
        match = re.fullmatch(r"\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*", value)
        if match is None:
            raise MetadataError(f"unparseable HLS bandpass tag for band {band}: {value!r}")
        parsed[role] = (float(match.group(1)), float(match.group(2)))
    return parsed
