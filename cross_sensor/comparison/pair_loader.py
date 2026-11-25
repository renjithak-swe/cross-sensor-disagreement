"""Reads the rasters and metadata one pair needs and turns them into reflectance and masks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..config import QualityConfig
from ..discovery.observation_discovery import Observation
from ..discovery.sentinel2_metadata import GranuleGeometry, parse_boa_scaling, parse_granule_geometry
from ..errors import GridAlignmentError, MetadataError
from ..harmonization.grid_alignment import GridDefinition, Reducer, block_reduce, replicate_to_grid
from ..harmonization.spectral_adjustment import parse_hls_bandpass_tags
from ..preprocessing.band_mapping import (
    HLS_L30,
    HLS_S30,
    LANDSAT_C2_L2,
    SENTINEL2_BANDS_READ,
    SENTINEL2_L2A,
    AuxiliaryRole,
    BandMapping,
    SpectralRole,
)
from ..preprocessing.quality_masking import (
    hls_high_aerosol,
    hls_invalid_from_fmask,
    landsat_invalid_from_qa,
    sentinel2_invalid_from_scl,
)
from ..preprocessing.reflectance_scaling import apply_scaling, file_scaling, landsat_scaling, sentinel2_scaling
from ..retrieval.raster_source import RasterWindow
from .comparison_run import AngleLayers, ComparisonGrids, HlsInputs, LandsatInputs, Layer, PairInputs, Sentinel2Inputs

LOGGER = logging.getLogger(__name__)


class AssetSource(Protocol):
    def read(self, href: str, bounds: tuple[float, float, float, float], bounds_crs: str) -> RasterWindow: ...

    def fetch_text(self, href: str) -> str: ...


@dataclass(frozen=True)
class HlsGranule:
    item_id: str
    assets: dict[str, str]


@dataclass
class PairProvenance:
    """Everything about a pair's inputs that is worth recording, beyond the item IDs."""

    sentinel2_sensing_time: datetime | None = None
    sentinel2_geometry: GranuleGeometry | None = None
    sentinel2_scaling: dict[str, dict[str, float]] = field(default_factory=dict)
    landsat_scaling: dict[str, dict[str, float]] = field(default_factory=dict)
    asset_hrefs: dict[str, str] = field(default_factory=dict)
    hls_sentinel2_id: str | None = None
    hls_landsat_id: str | None = None
    hls_source_check: str = ""
    hls_bandpass_tags: dict[str, list[float]] = field(default_factory=dict)


def _layer(window: RasterWindow, values: np.ndarray) -> Layer:
    return Layer(values=values, transform=window.transform)


class PairLoader:
    def __init__(self, source: AssetSource, grids: ComparisonGrids, quality: QualityConfig, crs: str) -> None:
        self._source = source
        self._grids = grids
        self._quality = quality
        self._crs = crs

    def _read(self, observation: Observation, mapping: BandMapping, role: SpectralRole | AuxiliaryRole, provenance: PairProvenance) -> RasterWindow:
        asset = observation.asset(mapping.asset_for(role))
        provenance.asset_hrefs[f"{observation.item_id}:{asset.key}"] = asset.href
        window = self._source.read(asset.href, self._grids.read_bounds, self._crs)
        if window.crs != self._crs:
            raise MetadataError(f"{observation.item_id}/{asset.key} is in {window.crs}, expected {self._crs}")
        return window

    def quality_overlap_fraction(self, sentinel2: Observation, landsat: Observation, provenance: PairProvenance) -> float:
        """Share of AOI cells usable by both quality masks, before any reflectance is read."""
        scl = self._read(sentinel2, SENTINEL2_L2A, AuxiliaryRole.SCENE_CLASSIFICATION, provenance)
        pixel_qa = self._read(landsat, LANDSAT_C2_L2, AuxiliaryRole.PIXEL_QA, provenance)
        saturation = self._read(landsat, LANDSAT_C2_L2, AuxiliaryRole.RADIOMETRIC_SATURATION, provenance)
        s2_invalid = sentinel2_invalid_from_scl(scl.array, self._quality.sentinel2_valid_scl_classes)
        landsat_invalid = landsat_invalid_from_qa(
            pixel_qa.array, saturation.array, self._quality.landsat_invalid_qa_bits, self._quality.landsat_saturation_bits
        )
        grids = self._grids
        s2_invalid_landsat = block_reduce(replicate_to_grid(s2_invalid, scl.transform, grids.fine), grids.fine, grids.landsat, Reducer.ANY)
        landsat_invalid_grid = replicate_to_grid(landsat_invalid, pixel_qa.transform, grids.landsat)
        usable = grids.inside_aoi_landsat & ~s2_invalid_landsat & ~landsat_invalid_grid
        return float(usable.sum()) / float(grids.inside_aoi_landsat.sum())

    def load(
        self,
        sentinel2: Observation,
        landsat: Observation,
        hls_sentinel2: HlsGranule | None,
        hls_landsat: HlsGranule | None,
        provenance: PairProvenance,
    ) -> PairInputs:
        product_xml = self._source.fetch_text(sentinel2.asset(SENTINEL2_L2A.asset_for(AuxiliaryRole.PRODUCT_METADATA)).href)
        granule_xml = self._source.fetch_text(sentinel2.asset(SENTINEL2_L2A.asset_for(AuxiliaryRole.GRANULE_METADATA)).href)
        boa_scaling = parse_boa_scaling(product_xml, SENTINEL2_BANDS_READ)
        geometry = parse_granule_geometry(granule_xml, ("B04", "B08", "B8A"))
        provenance.sentinel2_geometry = geometry
        provenance.sentinel2_sensing_time = geometry.sensing_time

        s2_layers: dict[SpectralRole, Layer] = {}
        for role in (SpectralRole.RED, SpectralRole.NIR_BROAD, SpectralRole.NIR_NARROW):
            band = SENTINEL2_L2A.asset_for(role)
            scaling = sentinel2_scaling(boa_scaling[band].quantification_value, boa_scaling[band].add_offset, band)
            provenance.sentinel2_scaling[band] = {"scale": scaling.scale, "offset": scaling.offset}
            window = self._read(sentinel2, SENTINEL2_L2A, role, provenance)
            s2_layers[role] = _layer(window, apply_scaling(window.array, scaling))
        scl = self._read(sentinel2, SENTINEL2_L2A, AuxiliaryRole.SCENE_CLASSIFICATION, provenance)
        sentinel2_inputs = Sentinel2Inputs(
            platform=sentinel2.platform,
            red=s2_layers[SpectralRole.RED],
            nir_broad=s2_layers[SpectralRole.NIR_BROAD],
            nir_narrow=s2_layers[SpectralRole.NIR_NARROW],
            is_invalid=_layer(scl, sentinel2_invalid_from_scl(scl.array, self._quality.sentinel2_valid_scl_classes)),
        )

        landsat_layers: dict[SpectralRole, Layer] = {}
        for role in (SpectralRole.RED, SpectralRole.NIR_NARROW):
            asset = landsat.asset(LANDSAT_C2_L2.asset_for(role))
            scaling = landsat_scaling(asset.raster_bands, asset.key)
            provenance.landsat_scaling[asset.key] = {"scale": scaling.scale, "offset": scaling.offset}
            window = self._read(landsat, LANDSAT_C2_L2, role, provenance)
            landsat_layers[role] = _layer(window, apply_scaling(window.array, scaling))
        pixel_qa = self._read(landsat, LANDSAT_C2_L2, AuxiliaryRole.PIXEL_QA, provenance)
        saturation = self._read(landsat, LANDSAT_C2_L2, AuxiliaryRole.RADIOMETRIC_SATURATION, provenance)
        landsat_inputs = LandsatInputs(
            red=landsat_layers[SpectralRole.RED],
            nir=landsat_layers[SpectralRole.NIR_NARROW],
            is_invalid=_layer(
                pixel_qa,
                landsat_invalid_from_qa(pixel_qa.array, saturation.array, self._quality.landsat_invalid_qa_bits, self._quality.landsat_saturation_bits),
            ),
        )

        hls_s30_inputs = hls_l30_inputs = None
        if hls_sentinel2 is not None and hls_landsat is not None:
            hls_s30_inputs, s30_tags = self._load_hls(hls_sentinel2, HLS_S30, provenance)
            hls_l30_inputs, l30_tags = self._load_hls(hls_landsat, HLS_L30, provenance)
            provenance.hls_sentinel2_id = hls_sentinel2.item_id
            provenance.hls_landsat_id = hls_landsat.item_id
            provenance.hls_source_check = _check_hls_sources(s30_tags, l30_tags, sentinel2, landsat)
            provenance.hls_bandpass_tags = {role.value: list(values) for role, values in parse_hls_bandpass_tags(s30_tags).items()}
            if provenance.hls_source_check != "ok":
                LOGGER.warning("HLS granules do not match the pair", extra={"detail": provenance.hls_source_check})
                hls_s30_inputs = hls_l30_inputs = None

        return PairInputs(
            crs=self._crs,
            sentinel2=sentinel2_inputs,
            landsat=landsat_inputs,
            hls_sentinel2=hls_s30_inputs,
            hls_landsat=hls_l30_inputs,
        )

    def _read_hls(self, granule: HlsGranule, asset_key: str, provenance: PairProvenance) -> RasterWindow:
        if asset_key not in granule.assets:
            raise MetadataError(f"{granule.item_id} has no asset {asset_key}")
        href = granule.assets[asset_key]
        provenance.asset_hrefs[f"{granule.item_id}:{asset_key}"] = href
        return self._source.read(href, self._grids.read_bounds, self._crs)

    def _on_mgrs(self, window: RasterWindow, values: np.ndarray) -> np.ndarray:
        mgrs: GridDefinition = self._grids.mgrs
        if abs(window.resolution - mgrs.resolution) > 1e-9:
            raise GridAlignmentError(f"HLS raster resolution {window.resolution} m is not {mgrs.resolution} m")
        return replicate_to_grid(values, window.transform, mgrs)

    def _load_hls(self, granule: HlsGranule, mapping: BandMapping, provenance: PairProvenance) -> tuple[HlsInputs, dict[str, str]]:
        reflectance: dict[SpectralRole, np.ndarray] = {}
        tags: dict[str, str] = {}
        for role in (SpectralRole.RED, SpectralRole.NIR_NARROW):
            window = self._read_hls(granule, mapping.asset_for(role), provenance)
            scaling = file_scaling(window.scale, window.offset, window.nodata, f"{granule.item_id}/{mapping.asset_for(role)}")
            reflectance[role] = self._on_mgrs(window, apply_scaling(window.array, scaling))
            tags.update(window.tags)
        fmask = self._read_hls(granule, mapping.asset_for(AuxiliaryRole.FMASK), provenance)
        is_invalid = self._on_mgrs(
            fmask, hls_invalid_from_fmask(fmask.array, self._quality.hls_invalid_fmask_bits, self._quality.hls_mask_high_aerosol)
        )
        is_high_aerosol = self._on_mgrs(fmask, hls_high_aerosol(fmask.array))
        angles: dict[AuxiliaryRole, Layer] = {}
        for role in (AuxiliaryRole.SUN_ZENITH, AuxiliaryRole.SUN_AZIMUTH, AuxiliaryRole.VIEW_ZENITH, AuxiliaryRole.VIEW_AZIMUTH):
            window = self._read_hls(granule, mapping.asset_for(role), provenance)
            if window.scale is None:
                raise MetadataError(f"{granule.item_id} angle layer declares no scale")
            values = window.array.astype(np.float32) * np.float32(window.scale)
            if window.nodata is not None:
                values[window.array == window.nodata] = np.nan
            angles[role] = _layer(window, values)
        return (
            HlsInputs(
                red=reflectance[SpectralRole.RED],
                nir=reflectance[SpectralRole.NIR_NARROW],
                is_invalid=is_invalid,
                is_high_aerosol=is_high_aerosol,
                angles=AngleLayers(
                    sun_zenith=angles[AuxiliaryRole.SUN_ZENITH],
                    sun_azimuth=angles[AuxiliaryRole.SUN_AZIMUTH],
                    view_zenith=angles[AuxiliaryRole.VIEW_ZENITH],
                    view_azimuth=angles[AuxiliaryRole.VIEW_AZIMUTH],
                ),
            ),
            tags,
        )


def _check_hls_sources(s30_tags: dict[str, Any], l30_tags: dict[str, Any], sentinel2: Observation, landsat: Observation) -> str:
    """Confirm from the HLS GeoTIFF tags that each granule was made from the same acquisition as the L2 item."""
    product_uri = str(s30_tags.get("PRODUCT_URI", ""))
    s2_stamp = sentinel2.acquired_at.strftime("%Y%m%dT%H%M%S")
    if s2_stamp not in product_uri or sentinel2.grid_id not in product_uri:
        return f"S30 PRODUCT_URI {product_uri!r} does not match {s2_stamp} T{sentinel2.grid_id}"
    landsat_product = str(l30_tags.get("LANDSAT_PRODUCT_ID", ""))
    path_row = landsat.grid_id.replace("/", "")
    landsat_date = landsat.acquired_at.strftime("%Y%m%d")
    if f"_{path_row}_{landsat_date}_" not in landsat_product:
        return f"L30 LANDSAT_PRODUCT_ID {landsat_product!r} does not match {path_row} {landsat_date}"
    return "ok"
