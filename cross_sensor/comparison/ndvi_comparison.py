"""Normalized Difference Vegetation Index from surface reflectance."""

from __future__ import annotations

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """(NIR - red) / (NIR + red); NaN where either input is NaN or the denominator is not positive."""
    if red.shape != nir.shape:
        raise ValueError("red and NIR arrays differ in shape")
    red64 = red.astype(np.float64)
    nir64 = nir.astype(np.float64)
    denominator = nir64 + red64
    is_defined = np.isfinite(red64) & np.isfinite(nir64) & (denominator > 0)
    ndvi = np.full(red.shape, np.nan, dtype=np.float64)
    np.divide(nir64 - red64, denominator, out=ndvi, where=is_defined)
    return ndvi.astype(np.float32)
