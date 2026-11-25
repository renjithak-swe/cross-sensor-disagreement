"""Comparison grids and exact moves between them.

Every grid here is north-up in one projected CRS. Exact operations (replication and block reduction) only work
between grids whose pixel edges fall on a common lattice; they raise GridAlignmentError otherwise instead of
silently interpolating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from ..errors import GridAlignmentError

ALIGNMENT_TOLERANCE_M = 1e-6


@dataclass(frozen=True)
class GridDefinition:
    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> float:
        return float(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return (left, top - self.height * self.resolution, left + self.width * self.resolution, top)

    def describe(self) -> dict[str, object]:
        return {
            "crs": self.crs,
            "resolution_m": self.resolution,
            "width": self.width,
            "height": self.height,
            "transform": list(self.transform)[:6],
            "bounds": list(self.bounds),
        }


class Reducer(str, Enum):
    MEAN = "mean"
    ALL = "all"
    ANY = "any"


def project_geometry(geometry: BaseGeometry, source_crs: str, target_crs: str) -> BaseGeometry:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def reproject_coordinates(coordinates: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geometry.segmentize(0.001), reproject_coordinates)


def build_aligned_grid(
    bounds: tuple[float, float, float, float],
    crs: str,
    resolution: float,
    origin_offset: float,
    is_inward: bool = False,
) -> GridDefinition:
    """Grid whose pixel edges sit at origin_offset modulo resolution; outward snapping covers the bounds."""
    min_x, min_y, max_x, max_y = bounds

    def snap(value: float, round_down: bool) -> float:
        steps = (value - origin_offset) / resolution
        return (math.floor(steps + 1e-9) if round_down else math.ceil(steps - 1e-9)) * resolution + origin_offset

    if is_inward:
        left, right = snap(min_x, False), snap(max_x, True)
        bottom, top = snap(min_y, False), snap(max_y, True)
    else:
        left, right = snap(min_x, True), snap(max_x, False)
        bottom, top = snap(min_y, True), snap(max_y, False)
    width = round((right - left) / resolution)
    height = round((top - bottom) / resolution)
    if width <= 0 or height <= 0:
        raise GridAlignmentError("bounds are too small for one grid cell")
    return GridDefinition(crs=crs, transform=Affine(resolution, 0.0, left, 0.0, -resolution, top), width=width, height=height)


def with_resolution(grid: GridDefinition, resolution: float) -> GridDefinition:
    """Same extent, finer cells. The coarse resolution must be a whole multiple of the new one."""
    factor = grid.resolution / resolution
    if abs(factor - round(factor)) > 1e-9 or round(factor) < 1:
        raise GridAlignmentError(f"{grid.resolution} m is not a multiple of {resolution} m")
    factor = round(factor)
    return GridDefinition(
        crs=grid.crs,
        transform=Affine(resolution, 0.0, grid.transform.c, 0.0, -resolution, grid.transform.f),
        width=grid.width * factor,
        height=grid.height * factor,
    )


def _check_north_up(transform: Affine, label: str) -> None:
    if transform.b != 0 or transform.d != 0 or transform.e != -transform.a or transform.a <= 0:
        raise GridAlignmentError(f"{label} is not a north-up square-pixel grid")


def _integer_ratio(numerator: float, denominator: float, label: str) -> int:
    ratio = numerator / denominator
    if abs(ratio - round(ratio)) > 1e-9:
        raise GridAlignmentError(f"{label}: {numerator} is not a whole multiple of {denominator}")
    return round(ratio)


def _lattice_offset(source_origin: float, target_origin: float, step: float, label: str) -> int:
    cells = (target_origin - source_origin) / step
    if abs(cells - round(cells)) * step > ALIGNMENT_TOLERANCE_M:
        raise GridAlignmentError(f"{label}: origins differ by {target_origin - source_origin} m, not a multiple of {step} m")
    return round(cells)


def replicate_to_grid(array: np.ndarray, array_transform: Affine, grid: GridDefinition) -> np.ndarray:
    """Copy each source pixel into every finer grid cell it contains (no interpolation)."""
    _check_north_up(array_transform, "source raster")
    _check_north_up(grid.transform, "target grid")
    source_resolution = float(array_transform.a)
    factor = _integer_ratio(source_resolution, grid.resolution, "replication")
    column_offset = _lattice_offset(array_transform.c, grid.transform.c, grid.resolution, "replication x")
    row_offset = _lattice_offset(grid.transform.f, array_transform.f, grid.resolution, "replication y")
    columns = (column_offset + np.arange(grid.width)) // factor
    rows = (row_offset + np.arange(grid.height)) // factor
    if column_offset < 0 or row_offset < 0 or columns[-1] >= array.shape[1] or rows[-1] >= array.shape[0]:
        raise GridAlignmentError("source raster does not cover the target grid")
    return array[np.ix_(rows, columns)]


def block_reduce(fine: np.ndarray, fine_grid: GridDefinition, coarse_grid: GridDefinition, reducer: Reducer) -> np.ndarray:
    """Reduce every block of fine cells that makes up one coarse cell. MEAN is area-weighted by construction."""
    if fine.shape != fine_grid.shape:
        raise GridAlignmentError("array shape does not match the fine grid")
    factor = _integer_ratio(coarse_grid.resolution, fine_grid.resolution, "block reduction")
    column_offset = _lattice_offset(fine_grid.transform.c, coarse_grid.transform.c, fine_grid.resolution, "block x")
    row_offset = _lattice_offset(coarse_grid.transform.f, fine_grid.transform.f, fine_grid.resolution, "block y")
    row_stop = row_offset + coarse_grid.height * factor
    column_stop = column_offset + coarse_grid.width * factor
    if row_offset < 0 or column_offset < 0 or row_stop > fine_grid.height or column_stop > fine_grid.width:
        raise GridAlignmentError("coarse grid extends beyond the fine grid")
    blocks = fine[row_offset:row_stop, column_offset:column_stop].reshape(coarse_grid.height, factor, coarse_grid.width, factor)
    if reducer is Reducer.MEAN:
        return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    if reducer is Reducer.ALL:
        return blocks.all(axis=(1, 3))
    return blocks.any(axis=(1, 3))


def coarsen_onto(values: np.ndarray, transform: Affine, target_transform: Affine, target_shape: tuple[int, int]) -> np.ndarray:
    """Mean of the finer native pixels that tile each pixel of a coarser native grid (e.g. B08 10 m onto B8A's 20 m).
    A coarse pixel is NaN if any of its fine pixels is NaN or lies outside ``values``."""
    _check_north_up(transform, "source raster")
    resolution = float(transform.a)
    factor = _integer_ratio(float(target_transform.a), resolution, "coarsening")
    column_start = _lattice_offset(transform.c, target_transform.c, resolution, "coarsening x")
    row_start = _lattice_offset(target_transform.f, transform.f, resolution, "coarsening y")
    height, width = target_shape[0] * factor, target_shape[1] * factor
    padded = np.full((height, width), np.nan, dtype=np.float32)
    rows = slice(max(0, row_start), min(values.shape[0], row_start + height))
    columns = slice(max(0, column_start), min(values.shape[1], column_start + width))
    if rows.stop > rows.start and columns.stop > columns.start:
        padded[rows.start - row_start:rows.stop - row_start, columns.start - column_start:columns.stop - column_start] = values[rows, columns]
    return padded.reshape(target_shape[0], factor, target_shape[1], factor).mean(axis=(1, 3))


def block_reduce_angle(fine_degrees: np.ndarray, fine_grid: GridDefinition, coarse_grid: GridDefinition) -> np.ndarray:
    """Circular mean of azimuths, so 359 deg and 1 deg average to 0 deg and not 180 deg."""
    radians = np.deg2rad(fine_degrees.astype(np.float64))
    mean_cos = block_reduce(np.cos(radians), fine_grid, coarse_grid, Reducer.MEAN)
    mean_sin = block_reduce(np.sin(radians), fine_grid, coarse_grid, Reducer.MEAN)
    return np.mod(np.rad2deg(np.arctan2(mean_sin, mean_cos)), 360.0).astype(np.float32)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square binary dilation; cells within ``radius`` cells of a True cell become True."""
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="edge")
    rows, columns = mask.shape
    grown = np.zeros_like(mask)
    for row_shift in range(2 * radius + 1):
        for column_shift in range(2 * radius + 1):
            grown |= padded[row_shift:row_shift + rows, column_shift:column_shift + columns]
    return grown


class GdalResampling(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


_KERNEL_SUPPORT_PIXELS = {GdalResampling.NEAREST: 0, GdalResampling.BILINEAR: 1, GdalResampling.CUBIC: 2}


def kernel_radius(method: GdalResampling, source_resolution: float, target_resolution: float) -> int:
    """GDAL widens bilinear and cubic kernels by the downsampling ratio; the mask grows by the same amount."""
    scale = max(1.0, target_resolution / source_resolution)
    return math.ceil(_KERNEL_SUPPORT_PIXELS[method] * scale)


def resample_with_gdal(
    values: np.ndarray,
    values_transform: Affine,
    crs: str,
    grid: GridDefinition,
    method: GdalResampling,
    is_invalid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """GDAL warp of one band onto ``grid``.

    Any target cell whose kernel touches an invalid source pixel is invalid, so masked pixels never leak into
    interpolated values.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if values.shape != is_invalid.shape:
        raise GridAlignmentError("values and mask differ in shape")
    filled = np.where(is_invalid, 0.0, values).astype(np.float32)
    destination = np.zeros(grid.shape, dtype=np.float32)
    reproject(
        source=filled,
        destination=destination,
        src_transform=values_transform,
        src_crs=crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=getattr(Resampling, method.value),
    )
    radius = kernel_radius(method, float(values_transform.a), grid.resolution)
    grown = dilate(is_invalid, radius).astype(np.uint8)
    invalid_destination = np.ones(grid.shape, dtype=np.uint8)
    reproject(
        source=grown,
        destination=invalid_destination,
        src_transform=values_transform,
        src_crs=crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    return destination, invalid_destination.astype(bool)


def rasterize_geometry(geometry: BaseGeometry, grid: GridDefinition) -> np.ndarray:
    """True for cells whose centre lies inside the geometry (geometry in the grid's CRS)."""
    from rasterio.features import geometry_mask

    return geometry_mask([geometry], out_shape=grid.shape, transform=grid.transform, invert=True, all_touched=False)
