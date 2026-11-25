"""Offline stand-ins for the STAC catalogue and the raster store, for tests and the ``demo`` command.

Synthetic rasters follow the real products' conventions: Sentinel-2 L2A digital numbers carry the +1000 offset
of processing baseline 04.00 and later, Landsat Collection 2 uses scale 2.75e-5 and offset -0.2 on a grid whose
pixel edges sit at 15 m modulo 30, and HLS uses scaled int16 reflectance on the MGRS lattice. The underlying
surface is a linear reflectance field, so the area-weighted mean of a Sentinel-2 block equals the value at the
Landsat cell centre and expected residuals are known exactly.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from .config import ExperimentConfig, parse_config
from .discovery.stac_client import StacItem
from .harmonization.grid_alignment import project_geometry
from .retrieval.raster_source import Bounds, RasterWindow

SYNTHETIC_CRS = "EPSG:32643"
SENTINEL2_GAIN = 1.03
SYNTHETIC_BBOX = (77.700, 12.950, 77.706, 12.956)
ACQUISITION = datetime(2024, 2, 27, 5, 7, 51, tzinfo=timezone.utc)
LANDSAT_ACQUISITION = datetime(2024, 2, 27, 5, 10, 55, tzinfo=timezone.utc)
CLOUD_SCL_CLASS = 9


def surface_reflectance(band: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear in x and y, so means over squares equal values at their centres."""
    x0, y0 = 785000.0, 1432000.0
    if band == "red":
        return 0.08 + 2e-5 * (x - x0) + 1e-5 * (y - y0)
    return 0.30 + 3e-5 * (x - x0) - 1e-5 * (y - y0)


ASSET_LATTICES: dict[str, tuple[float, float]] = {
    "B04": (10.0, 0.0), "B08": (10.0, 0.0), "B8A": (20.0, 0.0), "SCL": (20.0, 0.0),
    "red": (30.0, 15.0), "nir08": (30.0, 15.0), "qa_pixel": (30.0, 15.0), "qa_radsat": (30.0, 15.0),
    "HLS": (30.0, 0.0), "map": (10.0, 0.0),
}


def _window(bounds: Bounds, resolution: float, offset: float) -> tuple[Affine, int, int, np.ndarray, np.ndarray]:
    left = math.floor((bounds[0] - offset) / resolution) * resolution + offset
    right = math.ceil((bounds[2] - offset) / resolution) * resolution + offset
    bottom = math.floor((bounds[1] - offset) / resolution) * resolution + offset
    top = math.ceil((bounds[3] - offset) / resolution) * resolution + offset
    width, height = round((right - left) / resolution), round((top - bottom) / resolution)
    xs = left + (np.arange(width) + 0.5) * resolution
    ys = top - (np.arange(height) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(xs, ys)
    return Affine(resolution, 0, left, 0, -resolution, top), width, height, grid_x, grid_y


class SyntheticAssetSource:
    def __init__(self) -> None:
        self.reads: list[str] = []

    def read(self, href: str, bounds: Bounds, bounds_crs: str) -> RasterWindow:
        self.reads.append(href)
        _, item_id, asset = href.split("/", 3)[-3:]
        is_hls = item_id.startswith("HLS.")
        resolution, offset = ASSET_LATTICES["HLS" if is_hls else asset]
        transform, width, height, x, y = _window(bounds, resolution, offset)
        tags: dict[str, str] = {}
        scale = offset_value = nodata = None
        if is_hls:
            array, scale, nodata, tags = self._hls(item_id, asset, x, y)
        elif asset in ("B04", "B08", "B8A"):
            band = "red" if asset == "B04" else "nir"
            array = np.round(surface_reflectance(band, x, y) * SENTINEL2_GAIN * 10000 + 1000).astype(np.uint16)
            nodata = 0
        elif asset == "SCL":
            array = np.full((height, width), 4, dtype=np.uint8)
            array[:2, :2] = CLOUD_SCL_CLASS
        elif asset in ("red", "nir08"):
            band = "red" if asset == "red" else "nir"
            array = np.round((surface_reflectance(band, x, y) + 0.2) / 2.75e-5).astype(np.uint16)
            nodata = 0
        elif asset == "qa_pixel":
            array = np.full((height, width), 1 << 6, dtype=np.uint16)
        elif asset == "qa_radsat":
            array = np.zeros((height, width), dtype=np.uint16)
        elif asset == "map":
            array = np.where(x < x.mean(), 50, 40).astype(np.uint8)
            nodata = 0
        else:
            raise KeyError(f"synthetic source has no asset {asset}")
        return RasterWindow(array, transform, SYNTHETIC_CRS, nodata, scale, offset_value, tags, href)

    def _hls(self, item_id: str, asset: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float | None, float | None, dict[str, str]]:
        is_sentinel = ".S30." in item_id
        tags = (
            {
                "PRODUCT_URI": f"S2A_MSIL1C_{ACQUISITION:%Y%m%dT%H%M%S}_N0510_R019_T43PGQ_20240227T070615.SAFE",
                "MSI band 04 bandpass adjustment slope and offset": "0.976500, 0.000900",
                "MSI band 8a bandpass adjustment slope and offset": "0.998300, -0.000100",
            }
            if is_sentinel
            else {"LANDSAT_PRODUCT_ID": f"LC08_L1TP_144051_{LANDSAT_ACQUISITION:%Y%m%d}_{LANDSAT_ACQUISITION:%Y%m%d}_02_RT"}
        )
        if asset in ("B04", "B05", "B8A"):
            band = "red" if asset == "B04" else "nir"
            return np.round(surface_reflectance(band, x, y) * 10000).astype(np.int16), 0.0001, -9999, tags
        if asset == "Fmask":
            return np.full(x.shape, 0b01 << 6, dtype=np.uint8), 1.0, 255, tags
        angle = {"SZA": 32.0, "SAA": 130.0, "VZA": 5.0, "VAA": 103.0 if is_sentinel else 283.0}[asset]
        return np.full(x.shape, round(angle * 100), dtype=np.uint16), 0.01, 40000, tags

    def fetch_text(self, href: str) -> str:
        if href.endswith("product-metadata"):
            offsets = "".join(f'<BOA_ADD_OFFSET band_id="{index}">-1000</BOA_ADD_OFFSET>' for index in range(13))
            return f"<Level-2A_User_Product><BOA_QUANTIFICATION_VALUE unit=\"none\">10000</BOA_QUANTIFICATION_VALUE>{offsets}</Level-2A_User_Product>"
        if href.endswith("granule-metadata"):
            views = "".join(
                f'<Mean_Viewing_Incidence_Angle bandId="{index}"><ZENITH_ANGLE unit="deg">5.0</ZENITH_ANGLE><AZIMUTH_ANGLE unit="deg">103.0</AZIMUTH_ANGLE></Mean_Viewing_Incidence_Angle>'
                for index in range(13)
            )
            return (
                f"<Level-2A_Tile_ID><SENSING_TIME>{ACQUISITION:%Y-%m-%dT%H:%M:%S}.500Z</SENSING_TIME>"
                f"<Mean_Sun_Angle><ZENITH_ANGLE unit=\"deg\">32.0</ZENITH_ANGLE><AZIMUTH_ANGLE unit=\"deg\">130.0</AZIMUTH_ANGLE></Mean_Sun_Angle>"
                f"<Mean_Viewing_Incidence_Angle_List>{views}</Mean_Viewing_Incidence_Angle_List></Level-2A_Tile_ID>"
            )
        raise KeyError(f"synthetic source has no text asset {href}")


def _footprint(margin_degrees: float = 0.05) -> dict[str, Any]:
    west, south, east, north = SYNTHETIC_BBOX
    return {
        "type": "Polygon",
        "coordinates": [[[west - margin_degrees, south - margin_degrees], [east + margin_degrees, south - margin_degrees], [east + margin_degrees, north + margin_degrees], [west - margin_degrees, north + margin_degrees], [west - margin_degrees, south - margin_degrees]]],
    }


def _assets(item_id: str, keys: list[str], raster_bands: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    return {key: {"href": f"synthetic://{item_id}/{key}", **({"raster:bands": raster_bands[key]} if raster_bands and key in raster_bands else {})} for key in keys}


LANDSAT_RASTER_BANDS = {key: [{"scale": 2.75e-05, "offset": -0.2, "nodata": 0, "data_type": "uint16"}] for key in ("red", "nir08")}


def sentinel2_item(item_id: str, acquired: datetime, platform: str = "Sentinel-2A", cloud: float = 5.0) -> StacItem:
    return {
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "geometry": _footprint(),
        "properties": {
            "datetime": acquired.isoformat().replace("+00:00", "Z"),
            "platform": platform,
            "eo:cloud_cover": cloud,
            "s2:mgrs_tile": "43PGQ",
            "s2:processing_baseline": "05.10",
            "s2:mean_solar_zenith": 32.0,
            "s2:mean_solar_azimuth": 130.0,
            "proj:code": SYNTHETIC_CRS,
        },
        "assets": _assets(item_id, ["B04", "B08", "B8A", "SCL", "product-metadata", "granule-metadata"]),
    }


def landsat_item(item_id: str, acquired: datetime, platform: str = "landsat-8", category: str = "T1") -> StacItem:
    return {
        "id": item_id,
        "collection": "landsat-c2-l2",
        "geometry": _footprint(),
        "properties": {
            "datetime": acquired.isoformat().replace("+00:00", "Z"),
            "platform": platform,
            "eo:cloud_cover": 2.0,
            "landsat:wrs_path": "144",
            "landsat:wrs_row": "051",
            "landsat:correction": "L2SP",
            "landsat:collection_category": category,
            "view:sun_elevation": 58.0,
            "view:sun_azimuth": 130.0,
            "proj:code": SYNTHETIC_CRS,
        },
        "assets": _assets(item_id, ["red", "nir08", "qa_pixel", "qa_radsat"], LANDSAT_RASTER_BANDS),
    }


def hls_item(product: str, acquired: datetime) -> StacItem:
    item_id = f"HLS.{product}.T43PGQ.{acquired:%Y%jT%H%M%S}.v2.0"
    nir = "B8A" if product == "S30" else "B05"
    return {"id": item_id, "collection": f"hls2-{product.lower()}", "properties": {}, "assets": _assets(item_id, ["B04", nir, "Fmask", "SZA", "SAA", "VZA", "VAA"])}


class SyntheticCatalog:
    endpoint = "synthetic://catalog"

    def __init__(self) -> None:
        self._collections: dict[str, list[StacItem]] = {
            "sentinel-2-l2a": [
                sentinel2_item("S2A_SYNTH_PAIRED", ACQUISITION),
                sentinel2_item("S2B_SYNTH_FAR", ACQUISITION + timedelta(days=10), platform="Sentinel-2B"),
                sentinel2_item("S2A_SYNTH_CLOUDY", ACQUISITION + timedelta(days=5), cloud=95.0),
            ],
            "landsat-c2-l2": [
                landsat_item("LC08_SYNTH_PAIRED", LANDSAT_ACQUISITION),
                landsat_item("LE07_SYNTH_ETM", LANDSAT_ACQUISITION + timedelta(days=3), platform="landsat-7"),
                landsat_item("LC09_SYNTH_LONELY", LANDSAT_ACQUISITION + timedelta(days=25), platform="landsat-9"),
            ],
            "hls2-s30": [hls_item("S30", ACQUISITION)],
            "hls2-l30": [hls_item("L30", LANDSAT_ACQUISITION)],
            "esa-worldcover": [{"id": "WORLDCOVER_SYNTH", "properties": {}, "assets": _assets("WORLDCOVER_SYNTH", ["map"])}],
        }

    def search(self, collection: str, bbox: tuple[float, float, float, float], start: date, end: date) -> list[StacItem]:
        return list(self._collections.get(collection, []))

    def fetch_items(self, collection: str, item_ids: list[str]) -> list[StacItem]:
        wanted = set(item_ids)
        return [item for item in self._collections.get(collection, []) if item["id"] in wanted]


def synthetic_config(directory: Path, bootstrap_iterations: int = 200) -> ExperimentConfig:
    directory.mkdir(parents=True, exist_ok=True)
    west, south, east, north = SYNTHETIC_BBOX
    aoi_path = directory / "synthetic_aoi.geojson"
    aoi_path.write_text(
        json.dumps({"type": "Feature", "properties": {"name": "synthetic"}, "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}}),
        encoding="utf-8",
    )
    raw = {
        "catalog": {
            "endpoint": SyntheticCatalog.endpoint,
            "sentinel2_collection": "sentinel-2-l2a",
            "landsat_collection": "landsat-c2-l2",
            "hls_sentinel2_collection": "hls2-s30",
            "hls_landsat_collection": "hls2-l30",
            "landcover_collection": "esa-worldcover",
            "landcover_item_id": "WORLDCOVER_SYNTH",
        },
        "aoi": {"path": aoi_path.name},
        "period": {"start": "2024-01-01", "end": "2024-12-31"},
        "eligibility": {
            "sentinel2_platforms": ["Sentinel-2A", "Sentinel-2B", "Sentinel-2C"],
            "landsat_platforms": ["landsat-8", "landsat-9"],
            "landsat_collection_categories": ["T1"],
            "max_scene_cloud_cover_percent": 70.0,
            "max_time_difference_hours": 72.0,
            "min_valid_overlap_fraction": 0.25,
        },
        "grid": {"crs": SYNTHETIC_CRS, "resolution_m": 30.0, "landsat_origin_offset_m": 15.0, "mgrs_origin_offset_m": 0.0, "subcell_m": 5.0},
        "quality": {
            "sentinel2_valid_scl_classes": [4, 5, 6],
            "landsat_invalid_qa_bits": [0, 1, 2, 3, 4, 5],
            "landsat_saturation_bits": [3, 4],
            "hls_invalid_fmask_bits": [1, 2, 3, 4],
            "hls_mask_high_aerosol": False,
        },
        "landcover": {"min_purity": 0.75},
        "uncertainty": {"bootstrap_iterations": bootstrap_iterations, "random_seed": 7},
    }
    return parse_config(raw, directory)


def synthetic_aoi_projected_bounds() -> tuple[float, float, float, float]:
    import shapely.geometry

    return project_geometry(shapely.geometry.box(*SYNTHETIC_BBOX), "EPSG:4326", SYNTHETIC_CRS).bounds


def run_synthetic_demo(output_directory: Path, should_render_figures: bool = True) -> Any:
    from .runner import execute

    config = synthetic_config(output_directory / "config")
    return execute(config, SyntheticCatalog(), SyntheticAssetSource(), output_directory, should_render_figures=should_render_figures)
