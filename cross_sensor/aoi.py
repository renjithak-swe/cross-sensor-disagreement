"""Area of interest, read from a GeoJSON file in EPSG:4326."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import shapely.geometry
from shapely.geometry.base import BaseGeometry

from .errors import ConfigError
from .provenance.hashing import sha256_bytes


@dataclass(frozen=True)
class AreaOfInterest:
    name: str
    geometry: BaseGeometry
    source_path: Path
    sha256: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        min_x, min_y, max_x, max_y = self.geometry.bounds
        return (min_x, min_y, max_x, max_y)


def load_aoi(path: Path) -> AreaOfInterest:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read AOI file {path}: {error}") from error
    try:
        document = json.loads(content)
    except json.JSONDecodeError as error:
        raise ConfigError(f"AOI file {path} is not valid JSON: {error}") from error

    features = document.get("features") if document.get("type") == "FeatureCollection" else [document]
    if not isinstance(features, list) or len(features) != 1:
        raise ConfigError("AOI file must contain exactly one feature")
    feature = features[0]
    geometry_document = feature.get("geometry") if feature.get("type") == "Feature" else feature
    try:
        geometry = shapely.geometry.shape(geometry_document)
    except (AttributeError, TypeError, ValueError) as error:
        raise ConfigError(f"AOI geometry is invalid: {error}") from error
    if geometry.geom_type != "Polygon" or not geometry.is_valid or geometry.is_empty:
        raise ConfigError("AOI must be a single valid polygon")
    min_x, min_y, max_x, max_y = geometry.bounds
    if not (-180 <= min_x < max_x <= 180 and -90 <= min_y < max_y <= 90):
        raise ConfigError("AOI coordinates must be longitude/latitude in EPSG:4326")

    properties = feature.get("properties") or {}
    return AreaOfInterest(
        name=str(properties.get("name", path.stem)),
        geometry=geometry,
        source_path=path,
        sha256=sha256_bytes(content),
    )
