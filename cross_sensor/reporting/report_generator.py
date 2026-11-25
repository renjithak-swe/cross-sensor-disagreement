"""Writes run outputs: Parquet tables, GeoParquet footprints, COG rasters and the JSON summary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from ..comparison.comparison_run import VARIABLES, Condition
from ..metrics.residual_analyzer import HISTOGRAM_EDGES
from ..metrics.statistics_collector import ALL_CLASSES
from ..pipeline import RunResult


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def write_json(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(_clean(document), indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def write_observations(result: RunResult, directory: Path) -> list[Path]:
    import geopandas
    import pandas

    rows = []
    for record in result.observations:
        observation = record.observation
        rows.append(
            {
                "item_id": record.item_id,
                "sensor": record.sensor,
                "platform": observation.platform if observation else None,
                "acquired_at": observation.acquired_at if observation else None,
                "cloud_cover_percent": observation.cloud_cover_percent if observation else None,
                "grid_id": observation.grid_id if observation else None,
                "processing_version": observation.processing_version if observation else None,
                "collection_category": observation.collection_category if observation else None,
                "reason": record.decision.reason.value,
                "detail": record.decision.detail,
                "geometry": observation.footprint if observation else None,
            }
        )
    table = pandas.DataFrame(rows)
    observations_path = directory / "observations.parquet"
    table.drop(columns=["geometry"]).to_parquet(observations_path, index=False)
    footprints = geopandas.GeoDataFrame(table[table["geometry"].notna()], geometry="geometry", crs="EPSG:4326")
    aoi_row = geopandas.GeoDataFrame([{"item_id": "AOI", "sensor": "aoi", "reason": "", "geometry": result.aoi.geometry}], geometry="geometry", crs="EPSG:4326")
    footprints_path = directory / "footprints.geoparquet"
    pandas.concat([footprints, aoi_row], ignore_index=True).to_parquet(footprints_path, index=False)
    return [observations_path, footprints_path]


def write_pairs(result: RunResult, directory: Path) -> Path:
    import pandas

    rows = []
    for record in result.pairs:
        pair, provenance = record.pair, record.provenance
        s2 = pair.sentinel2
        geometry = provenance.sentinel2_geometry
        rows.append(
            {
                "pair_id": pair.pair_id,
                "reason": record.decision.reason.value,
                "detail": record.decision.detail,
                "landsat_item_id": pair.landsat.item_id,
                "landsat_platform": pair.landsat.platform,
                "landsat_acquired_at": pair.landsat.acquired_at,
                "landsat_cloud_cover_percent": pair.landsat.cloud_cover_percent,
                "landsat_collection_category": pair.landsat.collection_category,
                "landsat_sun_zenith_deg": pair.landsat.sun_zenith_deg,
                "landsat_sun_azimuth_deg": pair.landsat.sun_azimuth_deg,
                "sentinel2_item_id": s2.item_id if s2 else None,
                "sentinel2_platform": s2.platform if s2 else None,
                "sentinel2_acquired_at": s2.acquired_at if s2 else None,
                "sentinel2_tile_sensing_time": provenance.sentinel2_sensing_time,
                "sentinel2_processing_baseline": s2.processing_version if s2 else None,
                "sentinel2_cloud_cover_percent": s2.cloud_cover_percent if s2 else None,
                "sentinel2_sun_zenith_deg": geometry.sun_zenith_deg if geometry else (s2.sun_zenith_deg if s2 else None),
                "sentinel2_sun_azimuth_deg": geometry.sun_azimuth_deg if geometry else (s2.sun_azimuth_deg if s2 else None),
                "sentinel2_view_zenith_b04_deg": geometry.view_zenith_deg.get("B04") if geometry else None,
                "sentinel2_view_azimuth_b04_deg": geometry.view_azimuth_deg.get("B04") if geometry else None,
                "time_difference_hours": pair.time_difference_hours,
                "signed_time_difference_hours": pair.signed_time_difference_hours,
                "tile_signed_time_difference_hours": record.observed_signed_hours,
                "quality_overlap_fraction": record.quality_overlap_fraction,
                "valid_fraction": record.valid_fraction,
                "valid_landsat_grid_cells": record.valid_cells,
                "common_mask_sha256": record.common_mask_sha256,
                "hls_s30_item_id": provenance.hls_sentinel2_id,
                "hls_l30_item_id": provenance.hls_landsat_id,
                "hls_source_check": provenance.hls_source_check or None,
                "unavailable_conditions": json.dumps(record.unavailable_conditions) if record.unavailable_conditions else None,
                "angle_summary": json.dumps(record.angle_summary) if record.angle_summary else None,
                "sentinel2_scaling": json.dumps(provenance.sentinel2_scaling) if provenance.sentinel2_scaling else None,
                "landsat_scaling": json.dumps(provenance.landsat_scaling) if provenance.landsat_scaling else None,
                "asset_hrefs": json.dumps(provenance.asset_hrefs) if provenance.asset_hrefs else None,
            }
        )
    path = directory / "pairs.parquet"
    pandas.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_pair_metrics(result: RunResult, directory: Path) -> Path:
    import pandas

    path = directory / "pair_metrics.parquet"
    pandas.DataFrame(result.statistics.pair_rows).to_parquet(path, index=False)
    return path


def write_histograms(result: RunResult, directory: Path) -> Path:
    import pandas

    centres = (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:]) / 2
    frames = []
    for condition in Condition:
        for variable in VARIABLES:
            accumulator = result.statistics.pooled.get((condition, variable, ALL_CLASSES))
            if accumulator is None or accumulator.count == 0:
                continue
            nonzero = accumulator.histogram > 0
            frames.append(
                pandas.DataFrame(
                    {"condition": condition.value, "variable": variable, "bin_centre": centres[nonzero], "count": accumulator.histogram[nonzero]}
                )
            )
    path = directory / "residual_histograms.parquet"
    columns = {"condition": [], "variable": [], "bin_centre": [], "count": []}
    table = pandas.concat(frames, ignore_index=True) if frames else pandas.DataFrame(columns)
    table.to_parquet(path, index=False)
    return path


def write_cog(path: Path, bands: list[np.ndarray], transform: Affine, crs: str, descriptions: list[str], nodata: float | None) -> Path:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy as copy_dataset

    height, width = bands[0].shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(bands),
        "dtype": bands[0].dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with MemoryFile() as memory, memory.open(**profile) as dataset:
        for index, (band, description) in enumerate(zip(bands, descriptions, strict=True), start=1):
            dataset.write(band, index)
            dataset.set_band_description(index, description)
        copy_dataset(dataset, path, driver="COG", COMPRESS="DEFLATE", PREDICTOR="2" if np.issubdtype(bands[0].dtype, np.integer) else "3")
    with rasterio.open(path) as check:
        if check.count != len(bands):
            raise OSError(f"COG {path} was not written correctly")
    return path


def write_rasters(result: RunResult, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    grids = result.grids
    crs = grids.landsat.crs
    paths = [
        write_cog(
            directory / "landcover_landsat_grid.tif",
            [result.landcover_dominant_purity[0], result.landcover_dominant_purity[1], result.landcover.landsat.astype(np.float32), grids.inside_aoi_landsat.astype(np.float32)],
            grids.landsat.transform,
            crs,
            ["worldcover_dominant_class", "dominant_class_fraction", "pure_class_or_0", "inside_aoi"],
            None,
        )
    ]
    example = result.example
    if example is None:
        return paths
    paths.append(
        write_cog(directory / f"example_{example.pair_id}_sentinel2_native_ndvi.tif", [example.sentinel2_native_ndvi], example.sentinel2_native_transform, crs, ["sentinel2_b04_b08_ndvi_10m"], float("nan"))
    )
    paths.append(
        write_cog(directory / f"example_{example.pair_id}_landsat_native_ndvi.tif", [example.landsat_native_ndvi], example.landsat_native_transform, crs, ["landsat_ndvi_30m"], float("nan"))
    )
    common_bands = [example.landsat_c1_ndvi, example.sentinel2_c1_ndvi, example.sentinel2_c1_ndvi - example.landsat_c1_ndvi]
    descriptions = ["landsat_ndvi_C1", "sentinel2_ndvi_C1", "difference_C1_s2_minus_landsat"]
    if example.sentinel2_e2_ndvi is not None and example.landsat_e2_ndvi is not None:
        common_bands += [example.landsat_e2_ndvi, example.sentinel2_e2_ndvi, example.sentinel2_e2_ndvi - example.landsat_e2_ndvi]
        descriptions += ["landsat_ndvi_E2", "sentinel2_ndvi_E2", "difference_E2_s2_minus_landsat"]
    paths.append(
        write_cog(directory / f"example_{example.pair_id}_common_grid_ndvi.tif", [band.astype(np.float32) for band in common_bands], grids.landsat.transform, crs, descriptions, float("nan"))
    )
    return paths
