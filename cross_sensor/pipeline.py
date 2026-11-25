"""End-to-end run: discover, select, compare, summarise, write outputs and the manifest."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from .aoi import AreaOfInterest, load_aoi
from .comparison.comparison_run import (
    ComparisonGrids,
    Condition,
    LandcoverLayers,
    PairComparison,
    PairInputs,
    compare_pair,
)
from .comparison.ndvi_comparison import compute_ndvi
from .comparison.pair_loader import AssetSource, HlsGranule, PairLoader, PairProvenance
from .config import ExperimentConfig
from .discovery.observation_discovery import Observation, Sensor, parse_observation
from .discovery.stac_client import StacCatalog, StacItem
from .eligibility.comparison_eligibility import ComparisonEligibility, EligibilityDecision, ReasonCode
from .eligibility.observation_pair_selector import ObservationPair, ObservationPairSelector, deduplicate_acquisitions
from .errors import CatalogError, GridAlignmentError, MetadataError, RasterAccessError, UnsupportedPlatformError
from .harmonization.grid_alignment import (
    GridDefinition,
    build_aligned_grid,
    project_geometry,
    rasterize_geometry,
    replicate_to_grid,
    with_resolution,
)
from .metrics.statistics_collector import StatisticsCollector
from .preprocessing.band_mapping import LANDSAT_C2_L2, SENTINEL2_L2A
from .preprocessing.landcover import class_area_fractions, dominant_class

LOGGER = logging.getLogger(__name__)
READ_MARGIN_CELLS = 3
SAME_DAY_HOURS = 1.0


@dataclass
class ObservationRecord:
    sensor: str
    item_id: str
    observation: Observation | None
    decision: EligibilityDecision


@dataclass
class PairRecord:
    pair: ObservationPair
    decision: EligibilityDecision
    quality_overlap_fraction: float | None = None
    valid_fraction: float | None = None
    valid_cells: int | None = None
    provenance: PairProvenance = field(default_factory=PairProvenance)
    angle_summary: dict[str, float] = field(default_factory=dict)
    unavailable_conditions: dict[str, str] = field(default_factory=dict)
    common_mask_sha256: str | None = None

    @property
    def observed_signed_hours(self) -> float | None:
        """Sentinel-2 minus Landsat time over the tile. The S2 STAC datetime is the datatake start, which can be
        ~20 min before the tile is sensed, so the granule SENSING_TIME is used when it was read."""
        if self.provenance.sentinel2_sensing_time is None:
            return self.pair.signed_time_difference_hours
        return (self.provenance.sentinel2_sensing_time - self.pair.landsat.acquired_at).total_seconds() / 3600


@dataclass
class ExampleMaps:
    pair_id: str
    sentinel2_native_ndvi: np.ndarray
    sentinel2_native_transform: Affine
    landsat_native_ndvi: np.ndarray
    landsat_native_transform: Affine
    landsat_c1_ndvi: np.ndarray
    sentinel2_c1_ndvi: np.ndarray
    landsat_e2_ndvi: np.ndarray | None
    sentinel2_e2_ndvi: np.ndarray | None
    score: tuple[int, float]


@dataclass
class RunResult:
    aoi: AreaOfInterest
    grids: ComparisonGrids
    landcover: LandcoverLayers
    landcover_dominant_purity: np.ndarray
    landcover_fractions: dict[int, float]
    landcover_source: dict[str, str]
    observations: list[ObservationRecord]
    pairs: list[PairRecord]
    statistics: StatisticsCollector
    example: ExampleMaps | None
    hls_items: dict[str, StacItem]


def build_grids(aoi: AreaOfInterest, config: ExperimentConfig) -> ComparisonGrids:
    grid_config = config.grid
    projected = project_geometry(aoi.geometry, "EPSG:4326", grid_config.crs)
    landsat = build_aligned_grid(projected.bounds, grid_config.crs, grid_config.resolution_m, grid_config.landsat_origin_offset_m)
    mgrs = build_aligned_grid(landsat.bounds, grid_config.crs, grid_config.resolution_m, grid_config.mgrs_origin_offset_m, is_inward=True)
    fine = with_resolution(landsat, grid_config.subcell_m)
    margin = READ_MARGIN_CELLS * grid_config.resolution_m
    left, bottom, right, top = landsat.bounds
    return ComparisonGrids(
        landsat=landsat,
        mgrs=mgrs,
        fine=fine,
        inside_aoi_landsat=rasterize_geometry(projected, landsat),
        read_bounds=(left - margin, bottom - margin, right + margin, top + margin),
    )


def load_landcover(
    catalog: StacCatalog, source: AssetSource, config: ExperimentConfig, aoi: AreaOfInterest, grids: ComparisonGrids
) -> tuple[LandcoverLayers, np.ndarray, dict[int, float], dict[str, str]]:
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    items = catalog.fetch_items(config.catalog.landcover_collection, [config.catalog.landcover_item_id])
    if len(items) != 1:
        raise CatalogError(f"land-cover item {config.catalog.landcover_item_id} not found")
    href = items[0]["assets"]["map"]["href"]
    window = source.read(href, grids.read_bounds, config.grid.crs)
    fine_classes = np.zeros(grids.fine.shape, dtype=np.uint8)
    reproject(
        source=window.array,
        destination=fine_classes,
        src_transform=window.transform,
        src_crs=window.crs,
        dst_transform=grids.fine.transform,
        dst_crs=grids.fine.crs,
        resampling=Resampling.nearest,
        src_nodata=window.nodata,
        dst_nodata=0,
    )
    fine_classes = fine_classes.astype(np.int16)
    landsat_dominant = dominant_class(fine_classes, grids.fine, grids.landsat)
    mgrs_dominant = dominant_class(fine_classes, grids.fine, grids.mgrs)
    inside_fine = rasterize_geometry(project_geometry(aoi.geometry, "EPSG:4326", config.grid.crs), grids.fine)
    layers = LandcoverLayers(
        fine=fine_classes,
        landsat=landsat_dominant.pure_classes(config.landcover_min_purity),
        mgrs=mgrs_dominant.pure_classes(config.landcover_min_purity),
    )
    dominant_with_purity = np.stack([landsat_dominant.classes.astype(np.float32), landsat_dominant.purity])
    source_record = {"collection": config.catalog.landcover_collection, "item_id": config.catalog.landcover_item_id, "href": href}
    return layers, dominant_with_purity, class_area_fractions(fine_classes, inside_fine), source_record


def discover_observations(
    catalog: StacCatalog,
    config: ExperimentConfig,
    aoi: AreaOfInterest,
    pinned_item_ids: dict[str, list[str]] | None,
) -> tuple[list[Observation], list[Observation], list[ObservationRecord]]:
    eligibility = ComparisonEligibility(config.eligibility, aoi.geometry, config.grid.crs)
    records: list[ObservationRecord] = []
    accepted: dict[Sensor, list[Observation]] = {Sensor.SENTINEL2: [], Sensor.LANDSAT: []}
    sources = (
        (Sensor.SENTINEL2, config.catalog.sentinel2_collection, SENTINEL2_L2A),
        (Sensor.LANDSAT, config.catalog.landsat_collection, LANDSAT_C2_L2),
    )
    for sensor, collection, mapping in sources:
        if pinned_item_ids is None:
            items = catalog.search(collection, aoi.bbox, config.period_start, config.period_end)
        else:
            items = catalog.fetch_items(collection, pinned_item_ids.get(sensor.value, []))
        for item in items:
            item_id = str(item.get("id", "<missing id>"))
            try:
                observation = parse_observation(item, sensor)
            except MetadataError as error:
                records.append(ObservationRecord(sensor.value, item_id, None, EligibilityDecision(ReasonCode.INVALID_ITEM_METADATA, str(error))))
                continue
            decision = eligibility.evaluate(observation, mapping)
            records.append(ObservationRecord(sensor.value, item_id, observation, decision))
            if decision.is_accepted:
                accepted[sensor].append(observation)

    for sensor in accepted:
        kept, duplicates = deduplicate_acquisitions(accepted[sensor])
        accepted[sensor] = kept
        rejected_ids = {observation.item_id: decision for observation, decision in duplicates}
        for record in records:
            if record.item_id in rejected_ids:
                record.decision = rejected_ids[record.item_id]
    return accepted[Sensor.SENTINEL2], accepted[Sensor.LANDSAT], records


_HLS_ID = re.compile(r"^HLS\.(?P<product>[SL]30)\.T(?P<tile>[0-9A-Z]{5})\.(?P<stamp>\d{7}T\d{6})\.v2\.0$")


def index_hls_items(items: list[StacItem]) -> dict[str, StacItem]:
    index: dict[str, StacItem] = {}
    for item in items:
        match = _HLS_ID.match(str(item.get("id", "")))
        if match:
            index[f"{match['product']}.T{match['tile']}.{match['stamp']}"] = item
    return index


def find_hls_granule(index: dict[str, StacItem], product: str, tile: str, observation: Observation) -> HlsGranule | None:
    """HLS IDs carry the source acquisition's start time to the second; allow one second of rounding."""
    for offset_seconds in (0, -1, 1):
        stamp = (observation.acquired_at + timedelta(seconds=offset_seconds)).strftime("%Y%jT%H%M%S")
        item = index.get(f"{product}.T{tile}.{stamp}")
        if item is not None:
            return HlsGranule(item_id=item["id"], assets={key: asset["href"] for key, asset in item["assets"].items()})
    return None


def _example_score(record: PairRecord) -> tuple[int, float]:
    same_day = int(record.pair.time_difference_hours is not None and record.pair.time_difference_hours < SAME_DAY_HOURS)
    return (same_day, record.valid_fraction or 0.0)


def run_experiment(
    config: ExperimentConfig,
    catalog: StacCatalog,
    source: AssetSource,
    pinned_item_ids: dict[str, list[str]] | None = None,
) -> RunResult:
    aoi = load_aoi(config.aoi_path)
    grids = build_grids(aoi, config)
    LOGGER.info("grids built", extra={"landsat_grid": grids.landsat.describe(), "mgrs_grid": grids.mgrs.describe()})
    landcover, dominant_purity, landcover_fractions, landcover_source = load_landcover(catalog, source, config, aoi, grids)

    sentinel2, landsat, observation_records = discover_observations(catalog, config, aoi, pinned_item_ids)
    LOGGER.info("observations eligible", extra={"sentinel2": len(sentinel2), "landsat": len(landsat)})
    pairs = ObservationPairSelector(config.eligibility.max_time_difference_hours).select(landsat, sentinel2)

    if pinned_item_ids is None:
        hls_items = catalog.search(config.catalog.hls_sentinel2_collection, aoi.bbox, config.period_start, config.period_end)
        hls_items += catalog.search(config.catalog.hls_landsat_collection, aoi.bbox, config.period_start, config.period_end)
    else:
        hls_items = catalog.fetch_items(config.catalog.hls_sentinel2_collection, pinned_item_ids.get("hls_s30", []))
        hls_items += catalog.fetch_items(config.catalog.hls_landsat_collection, pinned_item_ids.get("hls_l30", []))
    hls_index = index_hls_items(hls_items)

    loader = PairLoader(source, grids, config.quality, config.grid.crs)
    statistics = StatisticsCollector()
    pair_records: list[PairRecord] = []
    example: ExampleMaps | None = None

    for pair in pairs:
        record = PairRecord(pair=pair, decision=pair.decision)
        pair_records.append(record)
        if not pair.decision.is_accepted or pair.sentinel2 is None:
            continue
        try:
            record.quality_overlap_fraction = loader.quality_overlap_fraction(pair.sentinel2, pair.landsat, record.provenance)
            if record.quality_overlap_fraction < config.eligibility.min_valid_overlap_fraction:
                record.decision = EligibilityDecision(
                    ReasonCode.INSUFFICIENT_VALID_OVERLAP, f"quality masks leave {record.quality_overlap_fraction:.1%} of the AOI"
                )
                continue
            tile = pair.sentinel2.grid_id
            hls_s30 = find_hls_granule(hls_index, "S30", tile, pair.sentinel2)
            hls_l30 = find_hls_granule(hls_index, "L30", tile, pair.landsat)
            inputs = loader.load(pair.sentinel2, pair.landsat, hls_s30, hls_l30, record.provenance)
            comparison = compare_pair(inputs, grids)
        except (RasterAccessError, CatalogError) as error:
            record.decision = EligibilityDecision(ReasonCode.RASTER_READ_FAILED, str(error))
            LOGGER.error("pair skipped", extra={"pair_id": pair.pair_id, "reason": str(error)})
            continue
        except (MetadataError, GridAlignmentError, UnsupportedPlatformError) as error:
            record.decision = EligibilityDecision(ReasonCode.INVALID_RASTER_METADATA, str(error))
            LOGGER.error("pair skipped", extra={"pair_id": pair.pair_id, "reason": str(error)})
            continue

        record.valid_fraction = comparison.valid_fraction
        record.valid_cells = comparison.valid_landsat_cells
        record.angle_summary = comparison.angle_summary
        record.common_mask_sha256 = comparison.common_mask_sha256
        record.unavailable_conditions = {condition.value: reason for condition, reason in comparison.unavailable.items()}
        if comparison.valid_fraction < config.eligibility.min_valid_overlap_fraction:
            record.decision = EligibilityDecision(
                ReasonCode.INSUFFICIENT_VALID_OVERLAP, f"{comparison.valid_fraction:.1%} of the AOI valid in every condition"
            )
            continue

        statistics.add_pair(pair.pair_id, comparison, landcover)
        score = _example_score(record)
        if example is None or score > example.score:
            example = _capture_example(pair.pair_id, inputs, comparison, score)
        LOGGER.info(
            "pair compared",
            extra={"pair_id": pair.pair_id, "valid_fraction": round(comparison.valid_fraction, 4), "hours": round(pair.time_difference_hours or 0, 2)},
        )
        del inputs, comparison

    return RunResult(
        aoi=aoi,
        grids=grids,
        landcover=landcover,
        landcover_dominant_purity=dominant_purity,
        landcover_fractions=landcover_fractions,
        landcover_source=landcover_source,
        observations=observation_records,
        pairs=pair_records,
        statistics=statistics,
        example=example,
        hls_items={item["id"]: item for item in hls_items},
    )


def _capture_example(pair_id: str, inputs: PairInputs, comparison: PairComparison, score: tuple[int, float]) -> ExampleMaps:
    """Keep the arrays the map figures need for the best pair seen so far (same-day first, then most valid area)."""
    s2 = inputs.sentinel2
    s2_grid = GridDefinition("", s2.red.transform, s2.red.values.shape[1], s2.red.values.shape[0])
    s2_invalid = replicate_to_grid(s2.is_invalid.values.astype(bool), s2.is_invalid.transform, s2_grid)
    landsat_invalid = inputs.landsat.is_invalid.values.astype(bool)
    c1 = comparison.conditions[Condition.C1_AREA_WEIGHTED]
    e2 = comparison.conditions.get(Condition.E2_VIEW_ANGLE_BANDPASS)
    return ExampleMaps(
        pair_id=pair_id,
        sentinel2_native_ndvi=np.where(s2_invalid, np.nan, compute_ndvi(s2.red.values, s2.nir_broad.values)).astype(np.float32),
        sentinel2_native_transform=s2.red.transform,
        landsat_native_ndvi=np.where(landsat_invalid, np.nan, compute_ndvi(inputs.landsat.red.values, inputs.landsat.nir.values)).astype(np.float32),
        landsat_native_transform=inputs.landsat.red.transform,
        landsat_c1_ndvi=c1.landsat["ndvi"],
        sentinel2_c1_ndvi=c1.sentinel2["ndvi"],
        landsat_e2_ndvi=None if e2 is None else e2.landsat["ndvi"],
        sentinel2_e2_ndvi=None if e2 is None else e2.sentinel2["ndvi"],
        score=score,
    )
