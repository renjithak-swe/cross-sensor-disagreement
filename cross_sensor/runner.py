"""Runs the experiment and writes every output plus the manifest that ties them to their sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .comparison.comparison_run import CONDITION_LABELS
from .comparison.pair_loader import AssetSource
from .config import ExperimentConfig
from .discovery.stac_client import StacCatalog
from .metrics.analysis import build_summary
from .pipeline import RunResult, run_experiment
from .provenance.experiment_manifest import (
    ExperimentManifest,
    build_experiment_id,
    collect_software_versions,
    now_utc,
    read_git_commit,
    strip_query_string,
)
from .provenance.hashing import sha256_json
from .reporting.figures import render_all_figures
from .reporting.report_generator import (
    write_histograms,
    write_json,
    write_observations,
    write_pair_metrics,
    write_pairs,
    write_rasters,
)

LOGGER = logging.getLogger(__name__)
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent


def _source_observations(result: RunResult) -> list[dict[str, Any]]:
    entries = []
    for record in result.pairs:
        if not record.decision.is_accepted or record.pair.sentinel2 is None:
            continue
        pair, provenance = record.pair, record.provenance
        entries.append(
            {
                "pair_id": pair.pair_id,
                "sentinel2": {
                    "collection": pair.sentinel2.collection,
                    "item_id": pair.sentinel2.item_id,
                    "acquired_at": pair.sentinel2.acquired_at.isoformat(),
                    "tile_sensing_time": provenance.sentinel2_sensing_time.isoformat() if provenance.sentinel2_sensing_time else None,
                },
                "landsat": {
                    "collection": pair.landsat.collection,
                    "item_id": pair.landsat.item_id,
                    "acquired_at": pair.landsat.acquired_at.isoformat(),
                },
                "hls_s30_item_id": provenance.hls_sentinel2_id,
                "hls_l30_item_id": provenance.hls_landsat_id,
                "assets": {key: strip_query_string(href) for key, href in sorted(provenance.asset_hrefs.items())},
            }
        )
    return entries


def rerender_figures(output_directory: Path) -> Path:
    """Redraw figures from a finished run's files and refresh the manifest's output hashes.

    Only drawing code runs; no data is fetched or recomputed.
    """
    import json

    from .provenance.experiment_manifest import OutputRecord
    from .provenance.hashing import sha256_file
    from .reporting.figures import load_summary

    manifest_path = output_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = sorted((output_directory / "rasters").glob("example_*_common_grid_ndvi.tif"))
    example_pair = examples[0].name.split("_")[1] if examples else None
    figure_paths = render_all_figures(output_directory, load_summary(output_directory), example_pair)
    kept = [entry for entry in manifest["outputs"] if not entry["path"].startswith("figures/")]
    refreshed = [
        OutputRecord(path.relative_to(output_directory).as_posix(), sha256_file(path), path.stat().st_size).__dict__
        for path in sorted(figure_paths)
    ]
    manifest["outputs"] = kept + refreshed
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def pinned_ids_from_manifest(manifest: dict[str, Any]) -> dict[str, list[str]]:
    observations = manifest["source_observations"]
    return {
        "sentinel2": sorted({entry["sentinel2"]["item_id"] for entry in observations}),
        "landsat": sorted({entry["landsat"]["item_id"] for entry in observations}),
        "hls_s30": sorted({entry["hls_s30_item_id"] for entry in observations if entry.get("hls_s30_item_id")}),
        "hls_l30": sorted({entry["hls_l30_item_id"] for entry in observations if entry.get("hls_l30_item_id")}),
    }


def execute(
    config: ExperimentConfig,
    catalog: StacCatalog,
    source: AssetSource,
    output_directory: Path,
    pinned_item_ids: dict[str, list[str]] | None = None,
    should_render_figures: bool = True,
) -> ExperimentManifest:
    executed_at = now_utc()
    output_directory.mkdir(parents=True, exist_ok=True)
    result = run_experiment(config, catalog, source, pinned_item_ids)

    summary = build_summary(result, config.bootstrap_iterations, config.random_seed)
    outputs = [
        *write_observations(result, output_directory),
        write_pairs(result, output_directory),
        write_pair_metrics(result, output_directory),
        write_histograms(result, output_directory),
        write_json(output_directory / "summary.json", summary),
        *write_rasters(result, output_directory / "rasters"),
    ]
    if should_render_figures:
        outputs += render_all_figures(output_directory, summary, result.example.pair_id if result.example else None)

    grids = result.grids
    manifest = ExperimentManifest(
        experiment_id=build_experiment_id(config.raw, executed_at),
        executed_at=executed_at.isoformat(),
        catalog_endpoint=catalog.endpoint,
        collections={
            "sentinel2": config.catalog.sentinel2_collection,
            "landsat": config.catalog.landsat_collection,
            "hls_s30": config.catalog.hls_sentinel2_collection,
            "hls_l30": config.catalog.hls_landsat_collection,
            "landcover": f"{config.catalog.landcover_collection}/{config.catalog.landcover_item_id}",
        },
        aoi={"path": config.aoi_path.name, "sha256": result.aoi.sha256, "bbox_wgs84": list(result.aoi.bbox)},
        config=config.raw,
        config_sha256=sha256_json(config.raw),
        grid={"landsat": grids.landsat.describe(), "mgrs": grids.mgrs.describe(), "fine": grids.fine.describe(), "read_bounds": list(grids.read_bounds)},
        resampling={
            "C1": "exact area-weighted mean of Sentinel-2 over each 30 m cell, via 5 m sub-cells",
            "C2": "GDAL bilinear (warp), Sentinel-2 10 m to the 30 m Landsat grid",
            "C3": "GDAL nearest neighbour (warp), Sentinel-2 10 m to the 30 m Landsat grid",
            "C4": "Sentinel-2 exact 3x3 mean; Landsat GDAL cubic (warp) to the 30 m MGRS grid",
            "angles": "HLS 30 m angle layers averaged onto the Landsat grid (circular mean for azimuths)",
            "landcover": "WorldCover nearest neighbour to 5 m, dominant class per 30 m cell",
            "masks": "a cell is valid only if every contributing source pixel is valid; GDAL kernels grow the mask by their support",
            "conditions": CONDITION_LABELS,
        },
        quality_masks=config.raw.get("quality", {}),
        software_versions=collect_software_versions(),
        git_commit=read_git_commit(PROJECT_DIRECTORY),
        source_observations=_source_observations(result),
    )
    manifest.record_outputs(output_directory, outputs)
    manifest.write(output_directory / "manifest.json")
    LOGGER.info("run complete", extra={"experiment_id": manifest.experiment_id, "compared_pairs": summary["counts"]["compared_pairs"]})
    return manifest
