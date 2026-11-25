"""View-angle normalization with the c-factor method (Roy et al. 2016), as used by HLS.

reflectance_normalized = c * reflectance_observed, where c is the ratio of the Ross-Thick/Li-Sparse-Reciprocal
BRDF model evaluated at the normalized and the observed geometry. The model parameters are the fixed global
values published for HLS (Ju et al. 2025, Table 4, derived from MODIS MCD43 by Roy et al. 2016).

This normalizes view-angle effects only. It is not a per-pixel BRDF inversion, and the prescribed solar zenith
used here (the mean of the two observations) differs from the modelled solar zenith HLS uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import UnsupportedPlatformError
from ..preprocessing.band_mapping import SpectralRole

HEIGHT_TO_BASE = 2.0
BASE_TO_RADIUS = 1.0
JU_2025_TABLE_4 = "Ju et al. (2025), RSE 324:114723, Table 4 (after Roy et al. 2016)"


@dataclass(frozen=True)
class BrdfParameters:
    isotropic: float
    geometric: float
    volumetric: float


BRDF_PARAMETERS: dict[SpectralRole, BrdfParameters] = {
    SpectralRole.RED: BrdfParameters(0.1690, 0.0227, 0.0574),
    SpectralRole.NIR_BROAD: BrdfParameters(0.3093, 0.0330, 0.1535),
    SpectralRole.NIR_NARROW: BrdfParameters(0.3093, 0.0330, 0.1535),
}


def _phase_angle_cosine(sun_zenith: np.ndarray, view_zenith: np.ndarray, relative_azimuth: np.ndarray) -> np.ndarray:
    return np.cos(sun_zenith) * np.cos(view_zenith) + np.sin(sun_zenith) * np.sin(view_zenith) * np.cos(relative_azimuth)


def ross_thick_kernel(sun_zenith: np.ndarray, view_zenith: np.ndarray, relative_azimuth: np.ndarray) -> np.ndarray:
    """Angles in radians. Relative azimuth is view minus solar azimuth; 0 is the backscatter (hotspot) side."""
    cos_phase = np.clip(_phase_angle_cosine(sun_zenith, view_zenith, relative_azimuth), -1.0, 1.0)
    phase = np.arccos(cos_phase)
    return ((np.pi / 2 - phase) * cos_phase + np.sin(phase)) / (np.cos(sun_zenith) + np.cos(view_zenith)) - np.pi / 4


def li_sparse_reciprocal_kernel(sun_zenith: np.ndarray, view_zenith: np.ndarray, relative_azimuth: np.ndarray) -> np.ndarray:
    """Li-Sparse reciprocal kernel with h/b = 2 and b/r = 1, as in the MODIS BRDF product (Lucht et al. 2000)."""
    sun_prime = np.arctan(BASE_TO_RADIUS * np.tan(sun_zenith))
    view_prime = np.arctan(BASE_TO_RADIUS * np.tan(view_zenith))
    tan_sun, tan_view = np.tan(sun_prime), np.tan(view_prime)
    sec_sun, sec_view = 1.0 / np.cos(sun_prime), 1.0 / np.cos(view_prime)
    cos_phase_prime = _phase_angle_cosine(sun_prime, view_prime, relative_azimuth)
    distance = np.sqrt(np.maximum(tan_sun**2 + tan_view**2 - 2 * tan_sun * tan_view * np.cos(relative_azimuth), 0.0))
    cos_t = HEIGHT_TO_BASE * np.sqrt(distance**2 + (tan_sun * tan_view * np.sin(relative_azimuth)) ** 2) / (sec_sun + sec_view)
    t = np.arccos(np.clip(cos_t, -1.0, 1.0))
    overlap = (t - np.sin(t) * np.cos(t)) * (sec_sun + sec_view) / np.pi
    return overlap - sec_sun - sec_view + 0.5 * (1 + cos_phase_prime) * sec_sun * sec_view


def _modelled_reflectance(parameters: BrdfParameters, sun_zenith: np.ndarray, view_zenith: np.ndarray, relative_azimuth: np.ndarray) -> np.ndarray:
    return (
        parameters.isotropic
        + parameters.geometric * li_sparse_reciprocal_kernel(sun_zenith, view_zenith, relative_azimuth)
        + parameters.volumetric * ross_thick_kernel(sun_zenith, view_zenith, relative_azimuth)
    )


def c_factor(
    role: SpectralRole,
    observed_sun_zenith_deg: np.ndarray,
    observed_view_zenith_deg: np.ndarray,
    observed_sun_azimuth_deg: np.ndarray,
    observed_view_azimuth_deg: np.ndarray,
    target_sun_zenith_deg: np.ndarray,
) -> np.ndarray:
    """Ratio of modelled reflectance at nadir view and the target sun zenith to that at the observed geometry."""
    if role not in BRDF_PARAMETERS:
        raise UnsupportedPlatformError(f"no BRDF parameters for {role.value}")
    parameters = BRDF_PARAMETERS[role]
    observed = _modelled_reflectance(
        parameters,
        np.deg2rad(observed_sun_zenith_deg),
        np.deg2rad(observed_view_zenith_deg),
        np.deg2rad(observed_view_azimuth_deg - observed_sun_azimuth_deg),
    )
    target_sun = np.deg2rad(target_sun_zenith_deg)
    normalized = _modelled_reflectance(parameters, target_sun, np.zeros_like(target_sun), np.zeros_like(target_sun))
    return (normalized / observed).astype(np.float32)
