"""Article figures, drawn only from files the run wrote. Each figure has a wide and a narrow (phone) version.

Figures sit on white card stock in both site themes, so one light rendering serves both. Before/after pairs use
orange (dashed) and blue (solid) with direct labels: the pair passes colour-vision-deficiency separation checks,
and line style carries the same distinction for readers who cannot rely on hue.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SURFACE = "#ffffff"
INK = "#1d231f"
INK_MUTED = "#4b5650"
RULE = "#d9ddd3"
MASKED = "#e6e8e2"
BEFORE = "#eb6834"
AFTER = "#2a78d6"
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
WIDE_INCHES = 7.6
NARROW_INCHES = 3.7
DPI = 100
WORLDCOVER_COLOURS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000",
    60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}
CONDITION_SHORT = {
    "A_native": "A  native pixels",
    "C1_area_weighted": "C1 common 30 m grid",
    "C2_bilinear": "C2 GDAL bilinear",
    "C3_nearest": "C3 GDAL nearest",
    "C4_hls_grid_cubic": "C4 HLS grid, cubic",
    "C4_hls_grid_cubic_hls_clear_cells": "C4 on HLS-clear cells",
    "D0_broad_nir_20m_support": "D0 + B08 at 20 m support",
    "D1_narrow_nir": "D1 + narrow NIR (B8A)",
    "D2_narrow_nir_bandpass": "D2 D1 + bandpass",
    "E0_broad_nir_view_angle": "E0 C1 + view angle",
    "E1_narrow_nir_view_angle": "E1 D1 + view angle",
    "E2_narrow_nir_view_angle_bandpass": "E2 E1 + bandpass",
    "F_hls_products": "F  HLS L30 vs S30",
}


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "cross-sensor-disagreement",
            "font.family": "sans-serif",
            "font.sans-serif": ["Overpass", "Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": RULE,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
        }
    )
    return plt


def _style_axes(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.grid(axis="both", color=RULE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _heading(fig: Any, title: str, subtitle: str, is_narrow: bool) -> None:
    """Title and subtitle laid out in inches from the top edge, wrapped to the figure width."""
    height = fig.get_figheight()
    title_lines = textwrap.wrap(title, 40 if is_narrow else 90)
    subtitle_lines = textwrap.wrap(subtitle, 56 if is_narrow else 120)
    top = 1 - 0.08 / height
    fig.text(0.02, top, "\n".join(title_lines), ha="left", va="top", fontsize=10.5 if is_narrow else 11.5, fontweight="bold", color=INK, linespacing=1.25)
    subtitle_top = top - (0.2 * len(title_lines) + 0.06) / height
    fig.text(0.02, subtitle_top, "\n".join(subtitle_lines), ha="left", va="top", fontsize=8.5, color=INK_MUTED, linespacing=1.3)


def _save(fig: Any, path: Path) -> Path:
    fig.savefig(path, format="svg", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _both_layouts(directory: Path, stem: str, draw: Callable[[bool], Any]) -> list[Path]:
    return [_save(draw(False), directory / f"{stem}.svg"), _save(draw(True), directory / f"{stem}-narrow.svg")]


def _number(value: float | None) -> float:
    """JSON stores undefined intervals as null; plotting needs NaN."""
    return float("nan") if value is None else float(value)


def _error_extent(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[np.ndarray]:
    """Error-bar half-widths; a missing interval draws no bar instead of failing."""
    return [np.nan_to_num(values - lower, nan=0.0), np.nan_to_num(upper - values, nan=0.0)]


def _read_raster(path: Path) -> tuple[np.ndarray, Any, list[str]]:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read(), dataset.transform, list(dataset.descriptions)


def _extent(transform: Any, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    left, top = transform.c, transform.f
    return (left / 1000, (left + shape[1] * transform.a) / 1000, (top + shape[0] * transform.e) / 1000, top / 1000)


def _map_axes(ax: Any, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(RULE)
    width = extent[1] - extent[0]
    bar_km = 2.0 if width > 8 else 0.5
    bar_left = extent[0] + 0.04 * width
    bar_y = extent[2] + 0.05 * (extent[3] - extent[2])
    ax.plot([bar_left, bar_left + bar_km], [bar_y, bar_y], color=INK, linewidth=2.5, solid_capstyle="butt")
    label = f"{bar_km:g} km" if bar_km >= 1 else f"{bar_km * 1000:.0f} m"
    ax.text(bar_left + bar_km / 2, bar_y + 0.03 * (extent[3] - extent[2]), label, ha="center", va="bottom", fontsize=8, color=INK)


def _ndvi_colormap() -> Any:
    from matplotlib import colormaps

    return colormaps["Greens"].with_extremes(bad=MASKED)


def _diverging_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("s2_minus_landsat", list(DIVERGING)).with_extremes(bad=MASKED)


def figure_aoi_landcover(directory: Path, results: Path, summary: dict[str, Any]) -> list[Path]:
    from matplotlib.patches import Patch

    plt = _matplotlib()
    bands, transform, _ = _read_raster(results / "rasters" / "landcover_landsat_grid.tif")
    dominant, inside = bands[0], bands[3] > 0
    fractions = summary["landcover"]["area_fraction"]
    name_to_code = {name: code for code, name in _worldcover_names().items()}
    present = sorted((name_to_code[name], name, value) for name, value in fractions.items() if name in name_to_code and value >= 0.001)
    rgb = np.ones((*dominant.shape, 3))
    for code, _, _ in present:
        rgb[dominant == code] = _hex_to_rgb(WORLDCOVER_COLOURS[code])
    rgb[~inside] = _hex_to_rgb(MASKED)
    extent = _extent(transform, dominant.shape)

    def draw(is_narrow: bool) -> Any:
        if is_narrow:
            fig = plt.figure(figsize=(NARROW_INCHES, 7.4), dpi=DPI)
            map_ax = fig.add_axes([0.04, 0.42, 0.92, 0.44])
            bar_ax = fig.add_axes([0.42, 0.05, 0.52, 0.3])
        else:
            fig = plt.figure(figsize=(WIDE_INCHES, 4.2), dpi=DPI)
            map_ax = fig.add_axes([0.02, 0.04, 0.5, 0.76])
            bar_ax = fig.add_axes([0.74, 0.12, 0.23, 0.62])
        _heading(fig, "The study area is mostly built-up land, trees and cropland", f"ESA WorldCover 2021 v200 · {summary['aoi']['area_km2']:.0f} km² in east Bengaluru", is_narrow)
        map_ax.imshow(rgb, extent=extent, interpolation="nearest")
        _map_axes(map_ax, extent)
        ordered = sorted(present, key=lambda entry: entry[2])
        bar_ax.barh(range(len(ordered)), [value * 100 for _, _, value in ordered], color=[WORLDCOVER_COLOURS[code] for code, _, _ in ordered], edgecolor=INK_MUTED, linewidth=0.5, height=0.7)
        bar_ax.set_yticks(range(len(ordered)), [name for _, name, _ in ordered], fontsize=8)
        for index, (_, _, value) in enumerate(ordered):
            bar_ax.text(value * 100 + 1, index, f"{value:.1%}", va="center", fontsize=8, color=INK)
        bar_ax.set_xlim(0, max(value for _, _, value in ordered) * 100 * 1.35)
        bar_ax.set_xlabel("share of AOI area (%)", fontsize=8)
        _style_axes(bar_ax)
        bar_ax.grid(axis="y", visible=False)
        map_ax.legend(handles=[Patch(facecolor=MASKED, edgecolor=RULE, label="outside AOI")], loc="upper right", fontsize=7)
        return fig

    return _both_layouts(directory, "fig01-aoi-landcover", draw)


def _worldcover_names() -> dict[int, str]:
    from ..preprocessing.landcover import WORLDCOVER_CLASSES

    return WORLDCOVER_CLASSES


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def figure_timeline(directory: Path, results: Path) -> list[Path]:
    import pandas

    plt = _matplotlib()
    observations = pandas.read_parquet(results / "observations.parquet")
    pairs = pandas.read_parquet(results / "pairs.parquet")
    compared = pairs[pairs["reason"] == "ACCEPTED"]
    rows = [
        ("Sentinel-2 L2A", observations[observations["sensor"] == "sentinel2"]),
        ("Landsat 8/9 L2", observations[(observations["sensor"] == "landsat") & observations["platform"].isin(["landsat-8", "landsat-9"])]),
    ]

    def draw(is_narrow: bool) -> Any:
        fig = plt.figure(figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 3.6 if is_narrow else 2.9), dpi=DPI)
        ax = fig.add_axes([0.25 if is_narrow else 0.15, 0.18, 0.72 if is_narrow else 0.82, 0.5])
        _heading(fig, "Usable pairs cluster in the dry season" if not is_narrow else "Usable pairs cluster\nin the dry season", f"{len(compared)} compared pairs; grey ticks are items rejected for cloud, coverage or platform", is_narrow)
        for index, (label, table) in enumerate(rows):
            y = 2 - index
            is_ok = table["reason"] == "ACCEPTED"
            ax.scatter(table.loc[~is_ok, "acquired_at"], np.full((~is_ok).sum(), y), marker="|", s=60, color=RULE, linewidths=1.2)
            ax.scatter(table.loc[is_ok, "acquired_at"], np.full(is_ok.sum(), y), marker="|", s=60, color=INK_MUTED, linewidths=1.2)
        ax.scatter(compared["landsat_acquired_at"], np.zeros(len(compared)), s=36, color=AFTER, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_yticks([2, 1, 0], [rows[0][0], rows[1][0], "Compared pairs"], fontsize=8)
        ax.set_ylim(-0.6, 2.6)
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        if is_narrow:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
        return fig

    return _both_layouts(directory, "fig02-acquisition-timeline", draw)


def figure_native_maps(directory: Path, results: Path, example_pair: str, pair_info: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    s2, s2_transform, _ = _read_raster(results / "rasters" / f"example_{example_pair}_sentinel2_native_ndvi.tif")
    landsat, landsat_transform, _ = _read_raster(results / "rasters" / f"example_{example_pair}_landsat_native_ndvi.tif")
    common, common_transform, _ = _read_raster(results / "rasters" / f"example_{example_pair}_common_grid_ndvi.tif")
    full = _extent(common_transform, common[0].shape)
    centre_x, centre_y = (full[0] + full[1]) / 2, (full[2] + full[3]) / 2
    zoom = (centre_x - 1.0, centre_x + 1.0, centre_y - 1.0, centre_y + 1.0)
    colormap = _ndvi_colormap()
    panels = [
        ("Sentinel-2, 10 m (B04, B08)", s2[0], s2_transform),
        ("Landsat, 30 m (red, NIR)", landsat[0], landsat_transform),
    ]

    def draw(is_narrow: bool) -> Any:
        if is_narrow:
            fig, axes = plt.subplots(4, 1, figsize=(NARROW_INCHES, 12.4), dpi=DPI, gridspec_kw={"hspace": 0.18})
            axes_order = [(axes[0], 0, full), (axes[1], 1, full), (axes[2], 0, zoom), (axes[3], 1, zoom)]
            fig.subplots_adjust(top=0.9, bottom=0.06, left=0.04, right=0.96)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(WIDE_INCHES, 7.4), dpi=DPI, gridspec_kw={"hspace": 0.14, "wspace": 0.04})
            axes_order = [(axes[0, 0], 0, full), (axes[0, 1], 1, full), (axes[1, 0], 0, zoom), (axes[1, 1], 1, zoom)]
            fig.subplots_adjust(top=0.86, bottom=0.1, left=0.02, right=0.98)
        _heading(fig, "Same place, same morning, two native pixel grids", f"Pair {example_pair}: {pair_info['dates']} · grey = masked by each product's own cloud/shadow flags", is_narrow)
        image = None
        for ax, index, extent in axes_order:
            title, values, transform = panels[index]
            image = ax.imshow(values, extent=_extent(transform, values.shape), cmap=colormap, vmin=0.0, vmax=0.9, interpolation="nearest")
            _map_axes(ax, extent)
            ax.set_title(title + ("" if extent == full else " · 2 × 2 km detail"), fontsize=8.5)
            if extent == full:
                ax.add_patch(_rectangle(zoom))
        colorbar_ax = fig.add_axes([0.3, 0.05 if not is_narrow else 0.035, 0.4, 0.012 if not is_narrow else 0.008])
        bar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
        bar.set_label("NDVI", fontsize=8)
        bar.outline.set_edgecolor(RULE)
        bar.ax.tick_params(labelsize=7)
        return fig

    return _both_layouts(directory, "fig03-native-ndvi", draw)


def _rectangle(extent: tuple[float, float, float, float]) -> Any:
    from matplotlib.patches import Rectangle

    return Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2], fill=False, edgecolor=INK, linewidth=1.2)


def figure_common_grid_scatter(directory: Path, results: Path, example_pair: str) -> list[Path]:
    plt = _matplotlib()
    common, _, descriptions = _read_raster(results / "rasters" / f"example_{example_pair}_common_grid_ndvi.tif")
    panels = [("C1: common 30 m grid, raw L2", common[0], common[1])]
    if "landsat_ndvi_E2" in descriptions:
        panels.append(("E2: + B8A, view angle, bandpass", common[3], common[4]))

    def draw(is_narrow: bool) -> Any:
        columns = 1 if is_narrow else len(panels)
        rows = len(panels) if is_narrow else 1
        fig, axes = plt.subplots(rows, columns, figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 3.7 * rows + 1.0 if is_narrow else 4.3), dpi=DPI, squeeze=False)
        fig.subplots_adjust(top=0.82 if not is_narrow else 0.86, bottom=0.12 if not is_narrow else 0.07, left=0.11 if not is_narrow else 0.2, right=0.97, wspace=0.28, hspace=0.35)
        _heading(fig, "Correlation is high either way; the offset is what changes", f"Pair {example_pair}: every valid 30 m cell, density-shaded", is_narrow)
        for ax, (title, landsat_values, s2_values) in zip(axes.ravel(), panels, strict=True):
            is_valid = np.isfinite(landsat_values) & np.isfinite(s2_values)
            x, y = landsat_values[is_valid], s2_values[is_valid]
            ax.hexbin(x, y, gridsize=70, extent=(-0.1, 0.9, -0.1, 0.9), cmap="Blues", mincnt=1, bins="log", linewidths=0)
            ax.plot([-0.1, 0.9], [-0.1, 0.9], color=INK, linewidth=1, linestyle=(0, (4, 3)))
            residual = y - x
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.03, 0.97, f"r = {r:.3f}\nmean bias = {residual.mean():+.3f}\nMAE = {np.abs(residual).mean():.3f}\nn = {x.size:,}", transform=ax.transAxes, va="top", fontsize=8, color=INK)
            ax.set_xlim(-0.1, 0.9)
            ax.set_ylim(-0.1, 0.9)
            ax.set_aspect("equal")
            ax.set_xlabel("Landsat NDVI", fontsize=8.5)
            ax.set_ylabel("Sentinel-2 NDVI", fontsize=8.5)
            ax.set_title(title, fontsize=8.5)
            ax.text(0.66, 0.56, "1:1", fontsize=8, color=INK_MUTED, ha="left", va="top")
            _style_axes(ax)
        return fig

    return _both_layouts(directory, "fig04-common-grid-scatter", draw)


def figure_difference_map(directory: Path, results: Path, example_pair: str) -> list[Path]:
    plt = _matplotlib()
    common, transform, descriptions = _read_raster(results / "rasters" / f"example_{example_pair}_common_grid_ndvi.tif")
    panels = [("C1: common grid, raw L2", common[2])]
    if "difference_E2_s2_minus_landsat" in descriptions:
        panels.append(("E2: after spectral + view-angle steps", common[5]))
    extent = _extent(transform, common[0].shape)
    colormap = _diverging_colormap()

    def draw(is_narrow: bool) -> Any:
        rows = len(panels) if is_narrow else 1
        columns = 1 if is_narrow else len(panels)
        fig, axes = plt.subplots(rows, columns, figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 3.3 * rows + 1.3 if is_narrow else 4.3), dpi=DPI, squeeze=False)
        fig.subplots_adjust(top=0.82 if not is_narrow else 0.86, bottom=0.2 if not is_narrow else 0.12, left=0.02, right=0.98, wspace=0.05, hspace=0.15)
        _heading(fig, "Where the two sensors disagree", f"Pair {example_pair}: Sentinel-2 minus Landsat NDVI · grey = masked", is_narrow)
        image = None
        for ax, (title, values) in zip(axes.ravel(), panels, strict=True):
            image = ax.imshow(values, extent=extent, cmap=colormap, vmin=-0.15, vmax=0.15, interpolation="nearest")
            _map_axes(ax, extent)
            ax.set_title(title, fontsize=8.5)
        colorbar_ax = fig.add_axes([0.25, 0.1 if not is_narrow else 0.05, 0.5, 0.025 if not is_narrow else 0.012])
        bar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal", extend="both")
        bar.set_label("Sentinel-2 lower  ←  NDVI difference  →  Sentinel-2 higher", fontsize=8)
        bar.outline.set_edgecolor(RULE)
        bar.ax.tick_params(labelsize=7)
        return fig

    return _both_layouts(directory, "fig05-difference-map", draw)


def _histogram_density(histograms: Any, condition: str, variable: str, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    rows = histograms[(histograms["condition"] == condition) & (histograms["variable"] == variable)]
    edges = np.arange(-0.3, 0.3 + bin_width, bin_width)
    counts, _ = np.histogram(rows["bin_centre"], bins=edges, weights=rows["count"])
    total = rows["count"].sum()
    return (edges[:-1] + edges[1:]) / 2, counts / total / bin_width


def _distribution_figure(directory: Path, results: Path, summary: dict[str, Any], stem: str, before: str, after: str, title: str, before_label: str, after_label: str) -> list[Path]:
    import pandas

    plt = _matplotlib()
    histograms = pandas.read_parquet(results / "residual_histograms.parquet")
    centres, before_density = _histogram_density(histograms, before, "ndvi", 0.005)
    _, after_density = _histogram_density(histograms, after, "ndvi", 0.005)
    before_stats = summary["pooled"][before]["ndvi"]
    after_stats = summary["pooled"][after]["ndvi"]

    def draw(is_narrow: bool) -> Any:
        fig = plt.figure(figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 4.4 if is_narrow else 3.6), dpi=DPI)
        ax = fig.add_axes([0.14 if is_narrow else 0.08, 0.14, 0.82 if is_narrow else 0.88, 0.6 if is_narrow else 0.64])
        _heading(fig, title, f"All compared pairs pooled · {after_stats['pixels']:,} cells after, {before_stats['pairs']} pairs", is_narrow)
        separator = "\n" if is_narrow else ": "
        ax.plot(centres, before_density, color=BEFORE, linewidth=2, linestyle=(0, (5, 3)),
                label=f"{before_label}{separator}MAE {before_stats['mae']['estimate']:.3f}, bias {before_stats['mean_bias']['estimate']:+.3f}")
        ax.plot(centres, after_density, color=AFTER, linewidth=2,
                label=f"{after_label}{separator}MAE {after_stats['mae']['estimate']:.3f}, bias {after_stats['mean_bias']['estimate']:+.3f}")
        ax.axvline(0, color=INK_MUTED, linewidth=0.8)
        ax.legend(loc="upper left", fontsize=7.5 if is_narrow else 8, handlelength=2.6, labelcolor=INK)
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0, max(before_density.max(), after_density.max()) * 1.45)
        ax.set_xlabel("Sentinel-2 minus Landsat NDVI", fontsize=8.5)
        ax.set_ylabel("density", fontsize=8.5)
        ax.tick_params(labelsize=8)
        _style_axes(ax)
        return fig

    return _both_layouts(directory, stem, draw)


def figure_ladder(directory: Path, summary: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    ladder = summary["ladder"]
    steps = ladder["steps"]
    labels = [CONDITION_SHORT[step["condition"]] for step in steps]
    metrics = [("mae", "NDVI mean absolute difference"), ("mean_bias", "NDVI mean bias (S2 − Landsat)")]

    def draw(is_narrow: bool) -> Any:
        rows = 2 if is_narrow else 1
        columns = 1 if is_narrow else 2
        fig, axes = plt.subplots(rows, columns, figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 7.0 if is_narrow else 3.9), dpi=DPI, squeeze=False)
        fig.subplots_adjust(top=0.78 if not is_narrow else 0.84, bottom=0.14 if not is_narrow else 0.08, left=0.3 if not is_narrow else 0.44, right=0.97, wspace=0.12, hspace=0.45)
        _heading(fig, "Each harmonization step, applied in turn", f"Same {len(ladder['pairs'])} pairs and pixels at every step · bars = 95% CI from resampling pairs", is_narrow)
        for index, (ax, (metric, axis_label)) in enumerate(zip(axes.ravel(), metrics, strict=True)):
            values = np.array([_number(step["ndvi"][metric]["estimate"]) for step in steps])
            lower = np.array([_number(step["ndvi"][metric]["ci_lower"]) for step in steps])
            upper = np.array([_number(step["ndvi"][metric]["ci_upper"]) for step in steps])
            positions = np.arange(len(steps))[::-1]
            ax.errorbar(values, positions, xerr=_error_extent(values, lower, upper), fmt="o", color=AFTER, ecolor=AFTER, elinewidth=1.5, capsize=0, markersize=6, markeredgecolor=SURFACE)
            ax.set_yticks(positions, labels if is_narrow or index == 0 else [""] * len(labels), fontsize=8)
            if metric == "mean_bias":
                ax.axvline(0, color=INK_MUTED, linewidth=0.8)
            ax.set_xlabel(axis_label, fontsize=8.5)
            ax.tick_params(axis="x", labelsize=8)
            _style_axes(ax)
            ax.grid(axis="y", visible=False)
        return fig

    return _both_layouts(directory, "fig07-harmonization-ladder", draw)


def figure_time_separation(directory: Path, summary: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    rows = summary["temporal"]["per_pair"]
    hours = np.array([row["hours"] for row in rows])
    mae = np.array([row["c1_ndvi_mae"] for row in rows])
    spearman = summary["temporal"]["spearman_hours_vs_c1_ndvi_mae"]

    def draw(is_narrow: bool) -> Any:
        fig = plt.figure(figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 4.2 if is_narrow else 3.6), dpi=DPI)
        ax = fig.add_axes([0.17 if is_narrow else 0.09, 0.14, 0.78 if is_narrow else 0.87, 0.6])
        _heading(
            fig,
            "Does a longer gap between overpasses mean more disagreement?",
            f"One dot per pair, condition C1 · Spearman ρ = {_format_interval(spearman)} (n = {spearman['pairs']})",
            is_narrow,
        )
        ax.scatter(hours, mae, s=42, color=AFTER, edgecolor=SURFACE, linewidth=1)
        ax.set_xlabel("time between acquisitions (hours)", fontsize=8.5)
        ax.set_ylabel("NDVI mean absolute difference", fontsize=8.5)
        ax.set_xlim(-3, 75)
        ax.set_xticks([0, 24, 48, 72])
        ax.set_ylim(0, max(mae) * 1.15 if mae.size else 1)
        ax.tick_params(labelsize=8)
        _style_axes(ax)
        return fig

    return _both_layouts(directory, "fig08-time-separation", draw)


def _format_interval(interval: dict[str, Any]) -> str:
    estimate, lower, upper = interval.get("estimate"), interval.get("ci_lower"), interval.get("ci_upper")
    if estimate is None:
        return "n/a"
    if lower is None or upper is None:
        return f"{estimate:+.2f}"
    return f"{estimate:+.2f} [{lower:+.2f}, {upper:+.2f}]"


def figure_landcover_breakdown(directory: Path, summary: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    before = summary["landcover_breakdown"]["C1_area_weighted"]
    after = summary["landcover_breakdown"].get("E2_narrow_nir_view_angle_bandpass") or summary["landcover_breakdown"]["D1_narrow_nir"]
    after_name = "E2" if summary["landcover_breakdown"].get("E2_narrow_nir_view_angle_bandpass") else "D1"
    classes = [name for name in before["classes"] if name in after["classes"]]
    classes.sort(key=lambda name: before["classes"][name]["mae"]["estimate"])

    def draw(is_narrow: bool) -> Any:
        fig = plt.figure(figsize=(NARROW_INCHES if is_narrow else WIDE_INCHES, 0.36 * len(classes) + 1.6), dpi=DPI)
        ax = fig.add_axes([0.4 if is_narrow else 0.24, 0.13, 0.55 if is_narrow else 0.72, 0.66])
        _heading(fig, "Disagreement depends on what is on the ground", f"NDVI MAE in 30 m cells at least 75% one WorldCover class · {after['pairs']} pairs", is_narrow)
        positions = np.arange(len(classes))
        for offset, (block, colour, marker, label) in enumerate(((before, BEFORE, "o", "C1 common grid"), (after, AFTER, "s", f"{after_name} harmonized"))):
            values = np.array([_number(block["classes"][name]["mae"]["estimate"]) for name in classes])
            lower = np.array([_number(block["classes"][name]["mae"]["ci_lower"]) for name in classes])
            upper = np.array([_number(block["classes"][name]["mae"]["ci_upper"]) for name in classes])
            ax.errorbar(values, positions + (0.15 if offset == 0 else -0.15), xerr=_error_extent(values, lower, upper), fmt=marker, color=colour, ecolor=colour, elinewidth=1.4, markersize=6, markeredgecolor=SURFACE, label=label, linestyle="none")
        ax.set_yticks(positions, classes, fontsize=8)
        ax.set_xlabel("NDVI mean absolute difference", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(loc="lower right", fontsize=8, handletextpad=0.3)
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        return fig

    return _both_layouts(directory, "fig09-landcover", draw)


def figure_pipeline(directory: Path) -> list[Path]:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt = _matplotlib()
    boxes = [
        ("Discover", "STAC search:\nS2 L2A, Landsat\nC2 L2, HLS v2.0,\nWorldCover"),
        ("Select", "Reason code for\nevery rejection;\nnearest S2 within\n72 h of Landsat"),
        ("Mask", "SCL, QA_PIXEL,\nsaturation; one\nshared set of\nvalid 30 m cells"),
        ("Compare", "A → C1 → D1\n→ E1 → E2;\nside branches\nC2–C4 and F"),
        ("Evidence", "Metrics, pair\nbootstrap, manifest\nwith item IDs\nand SHA-256"),
    ]

    def draw(is_narrow: bool) -> Any:
        if is_narrow:
            fig = plt.figure(figsize=(NARROW_INCHES, 8.8), dpi=DPI)
            positions = [(0.08, 0.76 - index * 0.17, 0.84, 0.13) for index in range(len(boxes))]
        else:
            fig = plt.figure(figsize=(WIDE_INCHES, 3.0), dpi=DPI)
            positions = [(0.012 + index * 0.199, 0.06, 0.172, 0.66) for index in range(len(boxes))]
        _heading(fig, "The experiment as a pipeline", "Each stage writes its decisions to disk", is_narrow)
        for index, ((title, body), (left, bottom, width, height)) in enumerate(zip(boxes, positions, strict=True)):
            is_key = title == "Compare"
            fig.patches.append(
                FancyBboxPatch((left, bottom), width, height, boxstyle="round,pad=0.004,rounding_size=0.01", transform=fig.transFigure,
                               facecolor="#e4eedd" if is_key else SURFACE, edgecolor=INK, linewidth=1.3)
            )
            if is_narrow:
                fig.text(left + 0.04, bottom + height / 2, title, fontsize=9.5, fontweight="bold", va="center", color=INK)
                fig.text(left + 0.3, bottom + height / 2, textwrap.fill(" ".join(body.split()), 28), fontsize=8, va="center", color=INK_MUTED, linespacing=1.3)
            else:
                fig.text(left + 0.012, bottom + height - 0.05, title, fontsize=9.5, fontweight="bold", va="top", color=INK)
                fig.text(left + 0.012, bottom + height - 0.17, body, fontsize=8, va="top", color=INK_MUTED, linespacing=1.35)
            if index < len(boxes) - 1:
                if is_narrow:
                    start, end = (0.5, bottom), (0.5, positions[index + 1][1] + positions[index + 1][3])
                else:
                    start, end = (left + width, bottom + height / 2), (positions[index + 1][0], bottom + height / 2)
                fig.patches.append(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=INK_MUTED, linewidth=1.2))
        return fig

    return _both_layouts(directory, "fig10-pipeline", draw)


def _pair_info(results: Path, pair_id: str) -> dict[str, Any]:
    import pandas

    pairs = pandas.read_parquet(results / "pairs.parquet")
    row = pairs[pairs["pair_id"] == pair_id].iloc[0]
    landsat_date = row["landsat_acquired_at"].strftime("%d %b %Y")
    s2_date = row["sentinel2_acquired_at"].strftime("%d %b %Y")
    dates = f"Landsat {landsat_date}, Sentinel-2 {s2_date}" if landsat_date != s2_date else f"both {landsat_date}"
    return {"dates": dates}


def render_all_figures(results: Path, summary: dict[str, Any], example_pair: str | None) -> list[Path]:
    directory = results / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths += figure_aoi_landcover(directory, results, summary)
    paths += figure_timeline(directory, results)
    if example_pair is not None:
        info = _pair_info(results, example_pair)
        paths += figure_native_maps(directory, results, example_pair, info)
        paths += figure_common_grid_scatter(directory, results, example_pair)
        paths += figure_difference_map(directory, results, example_pair)
    if "C1_area_weighted" in summary["pooled"]:
        paths += _distribution_figure(
            directory, results, summary, "fig06-residuals-spatial", "A_native", "C1_area_weighted",
            "Comparing like-sized pixels narrows the spread", "A: native pixels", "C1: common 30 m grid",
        )
    if "D2_narrow_nir_bandpass" in summary["pooled"]:
        paths += _distribution_figure(
            directory, results, summary, "fig07a-residuals-spectral", "C1_area_weighted", "D2_narrow_nir_bandpass",
            "Matching the NIR band shifts the whole distribution", "C1: S2 broad NIR (B08)", "D2: B8A + bandpass",
        )
    if summary["ladder"]["pairs"]:
        paths += figure_ladder(directory, summary)
    if summary["temporal"]["per_pair"]:
        paths += figure_time_separation(directory, summary)
    if summary["landcover_breakdown"]["C1_area_weighted"]["classes"]:
        paths += figure_landcover_breakdown(directory, summary)
    paths += figure_pipeline(directory)
    return paths


def load_summary(results: Path) -> dict[str, Any]:
    return json.loads((results / "summary.json").read_text(encoding="utf-8"))
