"""Turn raw STAC items into sensor-agnostic observation records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import shapely.geometry
from shapely.geometry.base import BaseGeometry

from ..errors import MetadataError
from .stac_client import StacItem


class Sensor(str, Enum):
    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"


@dataclass(frozen=True)
class AssetInfo:
    key: str
    href: str
    raster_bands: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Observation:
    sensor: Sensor
    item_id: str
    collection: str
    platform: str
    acquired_at: datetime
    cloud_cover_percent: float | None
    footprint: BaseGeometry
    crs: str | None
    grid_id: str
    processing_version: str
    collection_category: str | None
    sun_zenith_deg: float | None
    sun_azimuth_deg: float | None
    assets: dict[str, AssetInfo]

    def asset(self, key: str) -> AssetInfo:
        try:
            return self.assets[key]
        except KeyError as error:
            raise MetadataError(f"{self.item_id} has no asset '{key}'") from error


def parse_utc_datetime(value: Any, item_id: str) -> datetime:
    if not isinstance(value, str):
        raise MetadataError(f"{item_id}: datetime is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MetadataError(f"{item_id}: datetime '{value}' is not ISO 8601") from error
    if parsed.tzinfo is None:
        raise MetadataError(f"{item_id}: datetime '{value}' has no time zone")
    return parsed.astimezone(timezone.utc)


def _optional_float(properties: dict[str, Any], key: str, item_id: str) -> float | None:
    value = properties.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetadataError(f"{item_id}: property '{key}' is not a finite number")
    return float(value)


def _required_string(properties: dict[str, Any], key: str, item_id: str) -> str:
    value = properties.get(key)
    if not isinstance(value, str) or not value:
        raise MetadataError(f"{item_id}: property '{key}' is missing")
    return value


def _parse_assets(item: StacItem, item_id: str) -> dict[str, AssetInfo]:
    raw_assets = item.get("assets")
    if not isinstance(raw_assets, dict) or not raw_assets:
        raise MetadataError(f"{item_id}: item has no assets")
    assets: dict[str, AssetInfo] = {}
    for key, asset in raw_assets.items():
        if not isinstance(asset, dict) or not isinstance(asset.get("href"), str):
            raise MetadataError(f"{item_id}: asset '{key}' has no href")
        raster_bands = asset.get("raster:bands") or ()
        assets[key] = AssetInfo(key=key, href=asset["href"], raster_bands=tuple(raster_bands))
    return assets


def _parse_footprint(item: StacItem, item_id: str) -> BaseGeometry:
    try:
        footprint = shapely.geometry.shape(item["geometry"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise MetadataError(f"{item_id}: geometry is missing or invalid") from error
    if footprint.is_empty:
        raise MetadataError(f"{item_id}: geometry is empty")
    return footprint if footprint.is_valid else footprint.buffer(0)


def parse_observation(item: StacItem, sensor: Sensor) -> Observation:
    """Parse one STAC item. Raises MetadataError when a field the comparison needs is absent or invalid."""
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise MetadataError("item has no id")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise MetadataError(f"{item_id}: item has no properties")

    common = {
        "item_id": item_id,
        "collection": str(item.get("collection", "")),
        "platform": _required_string(properties, "platform", item_id),
        "acquired_at": parse_utc_datetime(properties.get("datetime"), item_id),
        "cloud_cover_percent": _optional_float(properties, "eo:cloud_cover", item_id),
        "footprint": _parse_footprint(item, item_id),
        "crs": properties.get("proj:code") or (f"EPSG:{properties['proj:epsg']}" if properties.get("proj:epsg") else None),
        "assets": _parse_assets(item, item_id),
    }

    if sensor is Sensor.SENTINEL2:
        return Observation(
            sensor=sensor,
            grid_id=_required_string(properties, "s2:mgrs_tile", item_id),
            processing_version=_required_string(properties, "s2:processing_baseline", item_id),
            collection_category=None,
            sun_zenith_deg=_optional_float(properties, "s2:mean_solar_zenith", item_id),
            sun_azimuth_deg=_optional_float(properties, "s2:mean_solar_azimuth", item_id),
            **common,
        )

    sun_elevation = _optional_float(properties, "view:sun_elevation", item_id)
    path = _required_string(properties, "landsat:wrs_path", item_id)
    row = _required_string(properties, "landsat:wrs_row", item_id)
    return Observation(
        sensor=sensor,
        grid_id=f"{path}/{row}",
        processing_version=_required_string(properties, "landsat:correction", item_id),
        collection_category=properties.get("landsat:collection_category"),
        sun_zenith_deg=None if sun_elevation is None else 90.0 - sun_elevation,
        sun_azimuth_deg=_optional_float(properties, "view:sun_azimuth", item_id),
        **common,
    )
