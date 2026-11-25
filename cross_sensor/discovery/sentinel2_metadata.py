"""Parsers for the Sentinel-2 L2A product and granule metadata XML files."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime

from ..errors import MetadataError
from .observation_discovery import parse_utc_datetime

SENTINEL2_BAND_INDEX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
    "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12,
}


@dataclass(frozen=True)
class BoaScaling:
    quantification_value: float
    add_offset: float


@dataclass(frozen=True)
class GranuleGeometry:
    sensing_time: datetime
    sun_zenith_deg: float
    sun_azimuth_deg: float
    view_zenith_deg: dict[str, float]
    view_azimuth_deg: dict[str, float]


def _parse_xml(text: str, label: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise MetadataError(f"{label} is not well-formed XML: {error}") from error


def _local_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _iter_local(root: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [element for element in root.iter() if _local_name(element) == name]


def _finite_float(text: str | None, label: str) -> float:
    try:
        value = float(text) if text is not None else math.nan
    except ValueError as error:
        raise MetadataError(f"{label} is not a number: {text!r}") from error
    if not math.isfinite(value):
        raise MetadataError(f"{label} is missing or not finite")
    return value


def parse_boa_scaling(product_xml: str, bands: tuple[str, ...]) -> dict[str, BoaScaling]:
    """Read BOA_QUANTIFICATION_VALUE and the per-band BOA_ADD_OFFSET.

    Processing baselines before 04.00 have no BOA_ADD_OFFSET element; ESA defines the offset as zero for them.
    """
    root = _parse_xml(product_xml, "product metadata")
    quantification_elements = _iter_local(root, "BOA_QUANTIFICATION_VALUE")
    if len(quantification_elements) != 1:
        raise MetadataError(f"expected one BOA_QUANTIFICATION_VALUE, found {len(quantification_elements)}")
    quantification = _finite_float(quantification_elements[0].text, "BOA_QUANTIFICATION_VALUE")
    if quantification <= 0:
        raise MetadataError("BOA_QUANTIFICATION_VALUE must be positive")

    offsets_by_index: dict[int, float] = {}
    for element in _iter_local(root, "BOA_ADD_OFFSET"):
        band_id = element.get("band_id")
        if band_id is None or not band_id.isdigit():
            raise MetadataError("BOA_ADD_OFFSET without a numeric band_id")
        offsets_by_index[int(band_id)] = _finite_float(element.text, f"BOA_ADD_OFFSET band {band_id}")

    scaling: dict[str, BoaScaling] = {}
    for band in bands:
        if band not in SENTINEL2_BAND_INDEX:
            raise MetadataError(f"unknown Sentinel-2 band {band}")
        index = SENTINEL2_BAND_INDEX[band]
        if offsets_by_index and index not in offsets_by_index:
            raise MetadataError(f"BOA_ADD_OFFSET is present for some bands but not for {band}")
        scaling[band] = BoaScaling(quantification_value=quantification, add_offset=offsets_by_index.get(index, 0.0))
    return scaling


def parse_granule_geometry(granule_xml: str, bands: tuple[str, ...]) -> GranuleGeometry:
    """Read the tile sensing time, the mean sun angles and the per-band mean viewing incidence angles."""
    root = _parse_xml(granule_xml, "granule metadata")
    sensing_elements = _iter_local(root, "SENSING_TIME")
    if not sensing_elements:
        raise MetadataError("granule metadata has no SENSING_TIME")
    sensing_time = parse_utc_datetime(sensing_elements[0].text, "granule metadata")

    mean_sun = _iter_local(root, "Mean_Sun_Angle")
    if len(mean_sun) != 1:
        raise MetadataError("granule metadata has no single Mean_Sun_Angle")
    sun_values = {_local_name(child): child.text for child in mean_sun[0]}

    view_zenith: dict[str, float] = {}
    view_azimuth: dict[str, float] = {}
    by_index = {element.get("bandId"): element for element in _iter_local(root, "Mean_Viewing_Incidence_Angle")}
    for band in bands:
        element = by_index.get(str(SENTINEL2_BAND_INDEX[band]))
        if element is None:
            raise MetadataError(f"granule metadata has no mean viewing angle for {band}")
        values = {_local_name(child): child.text for child in element}
        view_zenith[band] = _finite_float(values.get("ZENITH_ANGLE"), f"{band} view zenith")
        view_azimuth[band] = _finite_float(values.get("AZIMUTH_ANGLE"), f"{band} view azimuth")

    return GranuleGeometry(
        sensing_time=sensing_time,
        sun_zenith_deg=_finite_float(sun_values.get("ZENITH_ANGLE"), "mean sun zenith"),
        sun_azimuth_deg=_finite_float(sun_values.get("AZIMUTH_ANGLE"), "mean sun azimuth"),
        view_zenith_deg=view_zenith,
        view_azimuth_deg=view_azimuth,
    )
