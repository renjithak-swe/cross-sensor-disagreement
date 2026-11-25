"""Per-product cloud, shadow and invalid-pixel masks. Every function returns True where a pixel is unusable."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SCL_CLASS_NAMES = {
    0: "no data",
    1: "saturated or defective",
    2: "dark area / cast shadow",
    3: "cloud shadow",
    4: "vegetation",
    5: "not vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud medium probability",
    9: "cloud high probability",
    10: "thin cirrus",
    11: "snow or ice",
}
HLS_FMASK_FILL = 255
HLS_AEROSOL_SHIFT = 6
HLS_AEROSOL_HIGH = 0b11


def sentinel2_invalid_from_scl(scene_classification: np.ndarray, valid_classes: Iterable[int]) -> np.ndarray:
    return ~np.isin(scene_classification, np.fromiter(valid_classes, dtype=np.int64))


def _any_bit_set(values: np.ndarray, bits: Iterable[int]) -> np.ndarray:
    mask_value = 0
    for bit in bits:
        if not 0 <= bit < 16:
            raise ValueError(f"bit {bit} is outside a 16-bit word")
        mask_value |= 1 << bit
    return (values.astype(np.uint32) & mask_value) != 0


def landsat_invalid_from_qa(
    pixel_qa: np.ndarray,
    radiometric_saturation: np.ndarray,
    invalid_qa_bits: Iterable[int],
    saturation_bits: Iterable[int],
) -> np.ndarray:
    """QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
    QA_RADSAT bit n-1 flags saturation in band n (bits 3 and 4 are the red and NIR bands)."""
    if pixel_qa.shape != radiometric_saturation.shape:
        raise ValueError("QA_PIXEL and QA_RADSAT windows differ in shape")
    return _any_bit_set(pixel_qa, invalid_qa_bits) | _any_bit_set(radiometric_saturation, saturation_bits)


def hls_high_aerosol(fmask: np.ndarray) -> np.ndarray:
    return (fmask != HLS_FMASK_FILL) & (((fmask.astype(np.uint16) >> HLS_AEROSOL_SHIFT) & 0b11) == HLS_AEROSOL_HIGH)


def hls_invalid_from_fmask(fmask: np.ndarray, invalid_bits: Iterable[int], mask_high_aerosol: bool) -> np.ndarray:
    """HLS v2.0 Fmask: bit 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow, 4 snow/ice, 5 water,
    bits 6-7 aerosol level (11 = high); 255 is fill."""
    is_invalid = (fmask == HLS_FMASK_FILL) | _any_bit_set(fmask, invalid_bits)
    if mask_high_aerosol:
        is_invalid |= hls_high_aerosol(fmask)
    return is_invalid
