"""Explicit accept/reject decisions, each with a reason code. Nothing is dropped silently."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shapely.geometry.base import BaseGeometry

from ..config import EligibilityConfig
from ..discovery.observation_discovery import Observation, Sensor
from ..preprocessing.band_mapping import BandMapping


class ReasonCode(str, Enum):
    ACCEPTED = "ACCEPTED"
    INVALID_ITEM_METADATA = "INVALID_ITEM_METADATA"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
    UNSUPPORTED_COLLECTION_CATEGORY = "UNSUPPORTED_COLLECTION_CATEGORY"
    OUTSIDE_AOI = "OUTSIDE_AOI"
    AOI_NOT_FULLY_COVERED = "AOI_NOT_FULLY_COVERED"
    HIGH_CLOUD_COVER = "HIGH_CLOUD_COVER"
    REQUIRED_BAND_MISSING = "REQUIRED_BAND_MISSING"
    INVALID_RASTER_METADATA = "INVALID_RASTER_METADATA"
    DUPLICATE_ACQUISITION = "DUPLICATE_ACQUISITION"
    NO_COUNTERPART = "NO_COUNTERPART"
    EXCESSIVE_TIME_DIFFERENCE = "EXCESSIVE_TIME_DIFFERENCE"
    INSUFFICIENT_VALID_OVERLAP = "INSUFFICIENT_VALID_OVERLAP"
    RASTER_READ_FAILED = "RASTER_READ_FAILED"


@dataclass(frozen=True)
class EligibilityDecision:
    reason: ReasonCode
    detail: str = ""

    @property
    def is_accepted(self) -> bool:
        return self.reason is ReasonCode.ACCEPTED


ACCEPT = EligibilityDecision(ReasonCode.ACCEPTED)


class ComparisonEligibility:
    """Observation-level checks that need only catalogue metadata."""

    def __init__(self, config: EligibilityConfig, aoi_geometry: BaseGeometry, grid_crs: str) -> None:
        self._config = config
        self._aoi = aoi_geometry
        self._grid_crs = grid_crs

    def evaluate(self, observation: Observation, mapping: BandMapping) -> EligibilityDecision:
        allowed_platforms = (
            self._config.sentinel2_platforms if observation.sensor is Sensor.SENTINEL2 else self._config.landsat_platforms
        )
        if observation.platform not in allowed_platforms:
            return EligibilityDecision(ReasonCode.UNSUPPORTED_PLATFORM, observation.platform)
        if (
            observation.sensor is Sensor.LANDSAT
            and observation.collection_category not in self._config.landsat_collection_categories
        ):
            return EligibilityDecision(ReasonCode.UNSUPPORTED_COLLECTION_CATEGORY, str(observation.collection_category))
        if not observation.footprint.intersects(self._aoi):
            return EligibilityDecision(ReasonCode.OUTSIDE_AOI)
        if not observation.footprint.contains(self._aoi):
            covered = observation.footprint.intersection(self._aoi).area / self._aoi.area
            return EligibilityDecision(ReasonCode.AOI_NOT_FULLY_COVERED, f"{covered:.1%} of AOI covered")
        if observation.cloud_cover_percent is None:
            return EligibilityDecision(ReasonCode.INVALID_RASTER_METADATA, "eo:cloud_cover missing")
        if observation.cloud_cover_percent > self._config.max_scene_cloud_cover_percent:
            return EligibilityDecision(ReasonCode.HIGH_CLOUD_COVER, f"{observation.cloud_cover_percent:.1f}%")
        missing = mapping.missing_assets(observation.assets)
        if missing:
            return EligibilityDecision(ReasonCode.REQUIRED_BAND_MISSING, ",".join(missing))
        if observation.crs != self._grid_crs:
            return EligibilityDecision(ReasonCode.INVALID_RASTER_METADATA, f"CRS {observation.crs} is not {self._grid_crs}")
        return ACCEPT
