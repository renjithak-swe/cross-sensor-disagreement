"""Which asset holds which spectral or quality layer, per product."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class SpectralRole(str, Enum):
    RED = "red"
    NIR_BROAD = "nir_broad"
    NIR_NARROW = "nir_narrow"


class AuxiliaryRole(str, Enum):
    SCENE_CLASSIFICATION = "scene_classification"
    PIXEL_QA = "pixel_qa"
    RADIOMETRIC_SATURATION = "radiometric_saturation"
    FMASK = "fmask"
    PRODUCT_METADATA = "product_metadata"
    GRANULE_METADATA = "granule_metadata"
    SUN_ZENITH = "sun_zenith"
    SUN_AZIMUTH = "sun_azimuth"
    VIEW_ZENITH = "view_zenith"
    VIEW_AZIMUTH = "view_azimuth"


@dataclass(frozen=True)
class BandMapping:
    product: str
    spectral: dict[SpectralRole, str]
    auxiliary: dict[AuxiliaryRole, str]

    def asset_for(self, role: SpectralRole | AuxiliaryRole) -> str:
        table: dict = self.spectral if isinstance(role, SpectralRole) else self.auxiliary
        if role not in table:
            raise KeyError(f"{self.product} has no {role.value} layer")
        return table[role]

    def required_assets(self) -> set[str]:
        return set(self.spectral.values()) | set(self.auxiliary.values())

    def missing_assets(self, available: Iterable[str]) -> list[str]:
        return sorted(self.required_assets() - set(available))


SENTINEL2_L2A = BandMapping(
    product="Sentinel-2 MSI L2A",
    spectral={SpectralRole.RED: "B04", SpectralRole.NIR_BROAD: "B08", SpectralRole.NIR_NARROW: "B8A"},
    auxiliary={
        AuxiliaryRole.SCENE_CLASSIFICATION: "SCL",
        AuxiliaryRole.PRODUCT_METADATA: "product-metadata",
        AuxiliaryRole.GRANULE_METADATA: "granule-metadata",
    },
)

LANDSAT_C2_L2 = BandMapping(
    product="Landsat 8/9 OLI Collection 2 Level-2",
    spectral={SpectralRole.RED: "red", SpectralRole.NIR_NARROW: "nir08"},
    auxiliary={AuxiliaryRole.PIXEL_QA: "qa_pixel", AuxiliaryRole.RADIOMETRIC_SATURATION: "qa_radsat"},
)

_HLS_ANGLES = {
    AuxiliaryRole.SUN_ZENITH: "SZA",
    AuxiliaryRole.SUN_AZIMUTH: "SAA",
    AuxiliaryRole.VIEW_ZENITH: "VZA",
    AuxiliaryRole.VIEW_AZIMUTH: "VAA",
}

HLS_S30 = BandMapping(
    product="HLS v2.0 S30",
    spectral={SpectralRole.RED: "B04", SpectralRole.NIR_NARROW: "B8A"},
    auxiliary={AuxiliaryRole.FMASK: "Fmask", **_HLS_ANGLES},
)

HLS_L30 = BandMapping(
    product="HLS v2.0 L30",
    spectral={SpectralRole.RED: "B04", SpectralRole.NIR_NARROW: "B05"},
    auxiliary={AuxiliaryRole.FMASK: "Fmask", **_HLS_ANGLES},
)

SENTINEL2_BANDS_READ = ("B04", "B08", "B8A")
