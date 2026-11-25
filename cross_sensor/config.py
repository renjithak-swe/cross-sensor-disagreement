"""Experiment configuration, loaded from TOML and validated once at startup."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True)
class CatalogConfig:
    endpoint: str
    sentinel2_collection: str
    landsat_collection: str
    hls_sentinel2_collection: str
    hls_landsat_collection: str
    landcover_collection: str
    landcover_item_id: str


@dataclass(frozen=True)
class EligibilityConfig:
    sentinel2_platforms: tuple[str, ...]
    landsat_platforms: tuple[str, ...]
    landsat_collection_categories: tuple[str, ...]
    max_scene_cloud_cover_percent: float
    max_time_difference_hours: float
    min_valid_overlap_fraction: float


@dataclass(frozen=True)
class GridConfig:
    crs: str
    resolution_m: float
    landsat_origin_offset_m: float
    mgrs_origin_offset_m: float
    subcell_m: float


@dataclass(frozen=True)
class QualityConfig:
    sentinel2_valid_scl_classes: tuple[int, ...]
    landsat_invalid_qa_bits: tuple[int, ...]
    landsat_saturation_bits: tuple[int, ...]
    hls_invalid_fmask_bits: tuple[int, ...]
    hls_mask_high_aerosol: bool


@dataclass(frozen=True)
class ExperimentConfig:
    catalog: CatalogConfig
    aoi_path: Path
    period_start: date
    period_end: date
    eligibility: EligibilityConfig
    grid: GridConfig
    quality: QualityConfig
    landcover_min_purity: float
    bootstrap_iterations: int
    random_seed: int
    raw: dict[str, Any]


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] section")
    return value


def _require(section: dict[str, Any], section_name: str, key: str, expected: type | tuple[type, ...]) -> Any:
    if key not in section:
        raise ConfigError(f"missing {section_name}.{key}")
    value = section[key]
    if isinstance(value, bool) and expected in (int, float, (int, float)):
        raise ConfigError(f"{section_name}.{key} must be a number, got a boolean")
    if not isinstance(value, expected):
        raise ConfigError(f"{section_name}.{key} has type {type(value).__name__}")
    return value


def _number(section: dict[str, Any], section_name: str, key: str) -> float:
    return float(_require(section, section_name, key, (int, float)))


def _string_tuple(section: dict[str, Any], section_name: str, key: str) -> tuple[str, ...]:
    values = _require(section, section_name, key, list)
    if not values or not all(isinstance(value, str) for value in values):
        raise ConfigError(f"{section_name}.{key} must be a non-empty list of strings")
    return tuple(values)


def _int_tuple(section: dict[str, Any], section_name: str, key: str) -> tuple[int, ...]:
    values = _require(section, section_name, key, list)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ConfigError(f"{section_name}.{key} must be a list of integers")
    return tuple(values)


def parse_config(raw: dict[str, Any], base_directory: Path) -> ExperimentConfig:
    """Validate a parsed TOML document. Relative paths resolve against ``base_directory``."""
    catalog_raw = _section(raw, "catalog")
    catalog = CatalogConfig(
        **{key: _require(catalog_raw, "catalog", key, str) for key in CatalogConfig.__dataclass_fields__}
    )

    aoi_raw = _section(raw, "aoi")
    aoi_path = (base_directory / _require(aoi_raw, "aoi", "path", str)).resolve()

    period_raw = _section(raw, "period")
    try:
        period_start = date.fromisoformat(_require(period_raw, "period", "start", str))
        period_end = date.fromisoformat(_require(period_raw, "period", "end", str))
    except ValueError as error:
        raise ConfigError(f"period dates must be ISO dates: {error}") from error
    if period_end < period_start:
        raise ConfigError("period.end is before period.start")

    eligibility_raw = _section(raw, "eligibility")
    eligibility = EligibilityConfig(
        sentinel2_platforms=_string_tuple(eligibility_raw, "eligibility", "sentinel2_platforms"),
        landsat_platforms=_string_tuple(eligibility_raw, "eligibility", "landsat_platforms"),
        landsat_collection_categories=_string_tuple(eligibility_raw, "eligibility", "landsat_collection_categories"),
        max_scene_cloud_cover_percent=_number(eligibility_raw, "eligibility", "max_scene_cloud_cover_percent"),
        max_time_difference_hours=_number(eligibility_raw, "eligibility", "max_time_difference_hours"),
        min_valid_overlap_fraction=_number(eligibility_raw, "eligibility", "min_valid_overlap_fraction"),
    )
    if not 0.0 <= eligibility.min_valid_overlap_fraction <= 1.0:
        raise ConfigError("eligibility.min_valid_overlap_fraction must be between 0 and 1")
    if eligibility.max_time_difference_hours <= 0:
        raise ConfigError("eligibility.max_time_difference_hours must be positive")

    grid_raw = _section(raw, "grid")
    grid = GridConfig(
        crs=_require(grid_raw, "grid", "crs", str),
        resolution_m=_number(grid_raw, "grid", "resolution_m"),
        landsat_origin_offset_m=_number(grid_raw, "grid", "landsat_origin_offset_m"),
        mgrs_origin_offset_m=_number(grid_raw, "grid", "mgrs_origin_offset_m"),
        subcell_m=_number(grid_raw, "grid", "subcell_m"),
    )
    if grid.resolution_m <= 0 or grid.subcell_m <= 0:
        raise ConfigError("grid resolutions must be positive")
    if grid.resolution_m % grid.subcell_m:
        raise ConfigError("grid.resolution_m must be a multiple of grid.subcell_m")

    quality_raw = _section(raw, "quality")
    quality = QualityConfig(
        sentinel2_valid_scl_classes=_int_tuple(quality_raw, "quality", "sentinel2_valid_scl_classes"),
        landsat_invalid_qa_bits=_int_tuple(quality_raw, "quality", "landsat_invalid_qa_bits"),
        landsat_saturation_bits=_int_tuple(quality_raw, "quality", "landsat_saturation_bits"),
        hls_invalid_fmask_bits=_int_tuple(quality_raw, "quality", "hls_invalid_fmask_bits"),
        hls_mask_high_aerosol=_require(quality_raw, "quality", "hls_mask_high_aerosol", bool),
    )

    landcover_raw = _section(raw, "landcover")
    min_purity = _number(landcover_raw, "landcover", "min_purity")
    if not 0.0 < min_purity <= 1.0:
        raise ConfigError("landcover.min_purity must be in (0, 1]")

    uncertainty_raw = _section(raw, "uncertainty")
    iterations = _require(uncertainty_raw, "uncertainty", "bootstrap_iterations", int)
    if iterations < 100:
        raise ConfigError("uncertainty.bootstrap_iterations must be at least 100")

    return ExperimentConfig(
        catalog=catalog,
        aoi_path=aoi_path,
        period_start=period_start,
        period_end=period_end,
        eligibility=eligibility,
        grid=grid,
        quality=quality,
        landcover_min_purity=min_purity,
        bootstrap_iterations=iterations,
        random_seed=_require(uncertainty_raw, "uncertainty", "random_seed", int),
        raw=raw,
    )


def load_config(path: Path) -> ExperimentConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    return parse_config(raw, path.parent)
