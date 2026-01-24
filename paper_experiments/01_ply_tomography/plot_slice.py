#!/usr/bin/env python3
"""
Tomographic slice visualization (white-paper mode) for ECCV/CVPR figures.
Fixes: Correct aspect ratio for individual saved images by using square-pixel bins.
"""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from plyfile import PlyData

# ----------------------------------------------------------------------------- 
# Data locations (do not change)
# -----------------------------------------------------------------------------
BASELINE_PLY = "/home/ubuntu/lyj/Project/gaussian-splatting/output/170780ab-c/point_cloud/iteration_30000/point_cloud.ply"
OURS_PLY = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_DIR = "./paper_experiments/01_real_tomography/"

# ----------------------------------------------------------------------------- 
# Visualization configuration
# -----------------------------------------------------------------------------
SLICE_AXIS = "x"             # slice perpendicular to X
SLICE_RANGE = (-0.5, 0.5)    # keep the central 1m slab
VIEW_AXES = ("z", "y")       # histogram: X uses Z (depth), Y uses Y (height); Y flipped for CW rotation
BINS_Z = 1000                # Base resolution for Z axis

BACKGROUND = "#FFFFFF"
SPINE_COLOR = "#333333"
CMAP = plt.colormaps.get_cmap("turbo").copy()
CMAP.set_bad(color="white")  # masked (zero) areas are pure white


def set_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": "#111111",
        "axes.facecolor": BACKGROUND,
        "axes.titlepad": 12,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": BACKGROUND,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_ply_xyz_opacity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    verts = ply["vertex"].data
    required = ["x", "y", "z", "opacity"]
    for field in required:
        if field not in verts.dtype.names:
            raise KeyError(f"Field '{field}' not found in PLY: {path}")
    xyz = np.vstack([verts["x"], verts["y"], verts["z"]]).T.astype(np.float32)
    raw_opacity = np.asarray(verts["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    return xyz, opacity


def slice_points(
    xyz: np.ndarray,
    opacity: np.ndarray,
    slice_axis: str,
    slice_range: Tuple[float, float],
    view_axes: Tuple[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    view_idx0 = {"x": 0, "y": 1, "z": 2}[view_axes[0]]
    view_idx1 = {"x": 0, "y": 1, "z": 2}[view_axes[1]]
    mask = (xyz[:, axis_idx] >= slice_range[0]) & (xyz[:, axis_idx] <= slice_range[1])
    slice_xy = xyz[mask][:, [view_idx0, view_idx1]]
    # Clockwise rotation: (Y,Z) -> (Z,-Y) so bicycle lies horizontally
    slice_xy[:, 1] *= -1.0
    slice_w = opacity[mask]
    return slice_xy, slice_w


def compute_histogram(
    data_xy: np.ndarray,
    weights: np.ndarray,
    bins: List[int], # Changed to List[int] to support [nx, ny]
    hist_range: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, x_edges, y_edges = np.histogram2d(
        data_xy[:, 0],
        data_xy[:, 1],
        bins=bins,
        range=hist_range,
        weights=weights,
    )
    return hist, x_edges, y_edges


def style_axes(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    ax.set_title(title, color="#000000", fontweight="bold")
    ax.set_xlabel("Depth (Z, m)", color="#111111")
    if show_ylabel:
        ax.set_ylabel("Height (-Y, m)", color="#111111")
    ax.set_facecolor(BACKGROUND)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(1.2)
    ax.tick_params(axis="both", colors="#222222", length=4, width=1.0)


def main() -> None:
    set_matplotlib_style()
    ensure_output_dir(OUTPUT_DIR)

    print(f"Loading baseline from {BASELINE_PLY}")
    base_xyz, base_opacity = load_ply_xyz_opacity(BASELINE_PLY)

    print(f"Loading ours from {OURS_PLY}")
    ours_xyz, ours_opacity = load_ply_xyz_opacity(OURS_PLY)

    print("Slicing point clouds...")
    base_xy, base_w = slice_points(base_xyz, base_opacity, SLICE_AXIS, SLICE_RANGE, VIEW_AXES)
    ours_xy, ours_w = slice_points(ours_xyz, ours_opacity, SLICE_AXIS, SLICE_RANGE, VIEW_AXES)

    # Side view bounds after axis swap and Y flip: Z on x-axis, -Y on y-axis
    z_range = (-3.5, 3.5)
    y_range = (-3.0, 3.0)
    hist_range = [z_range, y_range]

    # --- 🔥 Fix Aspect Ratio: Calculate Square Pixel Bins 🔥 ---
    # To make imsave output match the physical proportions, 
    # the ratio of bins (ny/nx) must equal the ratio of physical lengths (Ly/Lx).
    z_span = z_range[1] - z_range[0]
    y_span = y_range[1] - y_range[0]
    
    bins_z = BINS_Z
    # Calculate Y bins proportional to the physical range
    bins_y = int(BINS_Z * (y_span / z_span))
    
    # [nx, ny]
    dynamic_bins = [bins_z, bins_y]
    
    print(f"Physical Span: Z={z_span:.1f}m, Y={y_span:.1f}m")
    print(f"Dynamic Bins:  Z={bins_z}, Y={bins_y} (Ensures square pixels for imsave)")

    print("Computing histograms...")
    H_base, z_edges, y_edges = compute_histogram(base_xy, base_w, bins=dynamic_bins, hist_range=hist_range)
    H_ours, _, _ = compute_histogram(ours_xy, ours_w, bins=dynamic_bins, hist_range=hist_range)

    # Mask zeros so empty space is pure white
    H_base[H_base == 0] = np.nan
    H_ours[H_ours == 0] = np.nan

    finite_vals = np.concatenate([
        H_base[np.isfinite(H_base)],
        H_ours[np.isfinite(H_ours)],
    ])
    positive_vals = finite_vals[finite_vals > 0]
    if positive_vals.size == 0:
        raise ValueError("No positive densities found in histograms.")
    vmin = positive_vals.min()
    vmax = np.percentile(positive_vals, 99.8)
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    print("Rendering figures...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)

    datasets = [("3DGS", H_base), ("Ours", H_ours)]
    extent = [z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]]

    for ax, (title, hist) in zip(axes, datasets):
        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=extent,
            cmap=CMAP,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        style_axes(ax, title, show_ylabel=(ax is axes[0]))

    fig.subplots_adjust(bottom=0.18, top=0.95, left=0.08, right=0.98, wspace=0.08)
    cax = fig.add_axes([0.22, 0.10, 0.56, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Opacity density (PowerNorm γ=0.5)", color="#111111", fontsize=12)
    cbar.ax.tick_params(labelsize=9, colors="#222222", length=3)
    cbar.outline.set_edgecolor("#555555")
    cbar.outline.set_linewidth(0.8)

    save_path = os.path.join(OUTPUT_DIR, "figure_1_tomography.png")
    print(f"Saving figure to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Save individual subplots (Pure data, Correct Aspect Ratio)
    # -------------------------------------------------------------------------
    print("Saving individual tomography maps (no axes, correct aspect ratio)...")
    for (name, hist) in datasets:
        filename = f"tomography_{name.lower()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # hist shape is (Z_bins, Y_bins) -> (1000, 857)
        # hist.T shape is (Y_bins, Z_bins) -> (857, 1000) -> Height, Width
        # Aspect Ratio = Width/Height = 1000/857 = 1.166
        # Physical Ratio = 7.0/6.0 = 1.166
        # Perfect match.
        image_data = np.flipud(hist.T)
        
        # Manual norm application + vmin/vmax for compatibility
        plt.imsave(
            filepath,
            norm(image_data),
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0
        )
        print(f"Saved {filepath}")

    print("Done.")


if __name__ == "__main__":
    main()