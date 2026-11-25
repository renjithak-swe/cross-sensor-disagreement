"""Raster and metadata retrieval: windowed reads of signed assets, with an optional local cache."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from affine import Affine

from ..errors import RasterAccessError
from ..provenance.experiment_manifest import strip_query_string
from ..provenance.hashing import sha256_bytes

LOGGER = logging.getLogger(__name__)

Bounds = tuple[float, float, float, float]
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3.0
HTTP_TIMEOUT_SECONDS = 60
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}


@dataclass(frozen=True)
class RasterWindow:
    array: np.ndarray
    transform: Affine
    crs: str
    nodata: float | None
    scale: float | None
    offset: float | None
    tags: dict[str, str]
    href: str

    @property
    def resolution(self) -> float:
        return float(self.transform.a)


class RasterSource(Protocol):
    def read(self, href: str, bounds: Bounds, bounds_crs: str) -> RasterWindow: ...


class TextSource(Protocol):
    def fetch_text(self, href: str) -> str: ...


def identity_signer(href: str) -> str:
    return href


def planetary_computer_signer(href: str) -> str:
    import planetary_computer

    return str(planetary_computer.sign(href))


def _cache_key(*parts: str) -> str:
    return sha256_bytes("|".join(parts).encode("utf-8"))[:40]


class SignedAssetSource:
    """Reads windows of cloud-optimised GeoTIFFs and small text assets, signing each URL just before use.

    Signing happens per request because Planetary Computer SAS tokens expire; unsigned hrefs are what gets
    cached and recorded, so no token reaches disk.
    """

    def __init__(self, signer: Callable[[str], str], cache_directory: Path | None = None) -> None:
        self._sign = signer
        self._cache_directory = cache_directory
        if cache_directory is not None:
            cache_directory.mkdir(parents=True, exist_ok=True)

    def read(self, href: str, bounds: Bounds, bounds_crs: str) -> RasterWindow:
        unsigned = strip_query_string(href)
        cache_path = self._cache_path("raster", unsigned, bounds_crs, ",".join(f"{value:.3f}" for value in bounds), suffix=".npz")
        if cache_path is not None and cache_path.exists():
            return _load_cached_window(cache_path)
        window = self._with_retries(lambda: self._read_remote(unsigned, bounds, bounds_crs), unsigned)
        if cache_path is not None:
            _save_cached_window(cache_path, window)
        return window

    def fetch_text(self, href: str) -> str:
        unsigned = strip_query_string(href)
        cache_path = self._cache_path("text", unsigned, suffix=".txt")
        if cache_path is not None and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        text = self._with_retries(lambda: self._fetch_remote_text(unsigned), unsigned)
        if cache_path is not None:
            _atomic_write_bytes(cache_path, text.encode("utf-8"))
        return text

    def _cache_path(self, kind: str, *parts: str, suffix: str) -> Path | None:
        if self._cache_directory is None:
            return None
        return self._cache_directory / kind / f"{_cache_key(*parts)}{suffix}"

    def _with_retries[T](self, operation: Callable[[], T], href: str) -> T:
        from rasterio.errors import RasterioError

        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return operation()
            except (OSError, urllib.error.URLError, RasterioError) as error:
                last_error = error
                LOGGER.warning("asset read failed", extra={"href": href, "attempt": attempt, "error": str(error)})
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)
        raise RasterAccessError(f"cannot read {href} after {MAX_ATTEMPTS} attempts: {last_error}")

    def _read_remote(self, unsigned_href: str, bounds: Bounds, bounds_crs: str) -> RasterWindow:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import Window

        with rasterio.Env(**GDAL_OPTIONS), rasterio.open(self._sign(unsigned_href)) as dataset:
            dataset_crs = dataset.crs.to_string() if dataset.crs else ""
            native_bounds = bounds if dataset_crs == bounds_crs else transform_bounds(bounds_crs, dataset.crs, *bounds, densify_pts=21)
            transform = dataset.transform
            column_start = math.floor((native_bounds[0] - transform.c) / transform.a)
            column_stop = math.ceil((native_bounds[2] - transform.c) / transform.a)
            row_start = math.floor((native_bounds[3] - transform.f) / transform.e)
            row_stop = math.ceil((native_bounds[1] - transform.f) / transform.e)
            if column_start < 0 or row_start < 0 or column_stop > dataset.width or row_stop > dataset.height:
                raise RasterAccessError(f"{unsigned_href} does not cover the requested bounds {bounds}")
            window = Window(column_start, row_start, column_stop - column_start, row_stop - row_start)
            array = dataset.read(1, window=window)
            return RasterWindow(
                array=array,
                transform=dataset.window_transform(window),
                crs=dataset_crs,
                nodata=dataset.nodata,
                scale=float(dataset.scales[0]) if dataset.scales else None,
                offset=float(dataset.offsets[0]) if dataset.offsets else None,
                tags={key: str(value) for key, value in dataset.tags().items()},
                href=unsigned_href,
            )

    def _fetch_remote_text(self, unsigned_href: str) -> str:
        request = urllib.request.Request(self._sign(unsigned_href), headers={"User-Agent": "cross-sensor-disagreement"})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _save_cached_window(path: Path, window: RasterWindow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "transform": list(window.transform)[:6],
        "crs": window.crs,
        "nodata": window.nodata,
        "scale": window.scale,
        "offset": window.offset,
        "tags": window.tags,
        "href": window.href,
    }
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, array=window.array, metadata=np.array(json.dumps(metadata)))
    temporary.replace(path)


def _load_cached_window(path: Path) -> RasterWindow:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        return RasterWindow(
            array=archive["array"],
            transform=Affine(*metadata["transform"]),
            crs=metadata["crs"],
            nodata=metadata["nodata"],
            scale=metadata["scale"],
            offset=metadata["offset"],
            tags=metadata["tags"],
            href=metadata["href"],
        )
