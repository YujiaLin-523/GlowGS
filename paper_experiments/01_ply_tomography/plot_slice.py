#!/usr/bin/env python3
"""
Fig 3: Opacity Tomography (Final Fix for Linux/Server)
"""

import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plyfile import PlyData

# ----------------------------------------------------------------------------- 
# Data locations
# -----------------------------------------------------------------------------
BASELINE_PLY = "/home/ubuntu/lyj/Project/gaussian-splatting/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OURS_PLY = "/home/ubuntu/lyj/Project/GlowGS/output/legacy_output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_DIR = "./paper_experiments/01_ply_tomography/"

# ----------------------------------------------------------------------------- 
# Visualization configuration
# -----------------------------------------------------------------------------
SLICE_AXIS = "x"             
SLICE_RANGE = (-0.5, 0.5)    
VIEW_AXES = ("z", "y")       
BINS_Z = 600                 # <--- 修改：稍微降低分辨率，让云图看起来更"实"
CROP_Y = (-2.8, 1.5)         # <--- 修改：裁剪 Y 轴，去掉顶部的空白天空
CROP_Z = (-3.2, 3.2)         # <--- 修改：裁剪 Z 轴

# Color Scheme
BACKGROUND = "#FFFFFF"
SPINE_COLOR = "#333333"
CMAP = plt.colormaps.get_cmap("turbo").copy()
CMAP.set_bad(color="white") 

def set_paper_style() -> None:
    """Robust Font Loading for Linux Servers"""
    # 尝试找到 Arial，如果找不到，自动回退到 DejaVu Sans (Linux 标配)
    font_names = [f.name for f in fm.fontManager.ttflist]
    if 'Arial' in font_names:
        font_family = 'Arial'
    elif 'Helvetica' in font_names:
        font_family = 'Helvetica'
    else:
        font_family = 'DejaVu Sans' # Safe fallback
    
    print(f"Using Font: {font_family}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family], 
        "font.size": 12,           # 稍微调大字号
        "axes.linewidth": 1.2,     # 边框稍微加粗
        "axes.edgecolor": SPINE_COLOR,
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_ply_xyz_opacity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    verts = ply["vertex"].data
    xyz = np.vstack([verts["x"], verts["y"], verts["z"]]).T.astype(np.float32)
    raw_opacity = np.asarray(verts["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    return xyz, opacity

def slice_points(xyz, opacity, slice_axis, slice_range, view_axes):
    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    view_idx0 = {"x": 0, "y": 1, "z": 2}[view_axes[0]]
    view_idx1 = {"x": 0, "y": 1, "z": 2}[view_axes[1]]
    
    mask = (xyz[:, axis_idx] >= slice_range[0]) & (xyz[:, axis_idx] <= slice_range[1])
    slice_xy = xyz[mask][:, [view_idx0, view_idx1]]
    slice_xy[:, 1] *= -1.0
    slice_w = opacity[mask]
    return slice_xy, slice_w

def compute_histogram(data_xy, weights, bins, hist_range):
    hist, x_edges, y_edges = np.histogram2d(
        data_xy[:, 0], data_xy[:, 1],
        bins=bins, range=hist_range, weights=weights,
    )
    return hist, x_edges, y_edges

def main() -> None:
    set_paper_style()
    ensure_output_dir(OUTPUT_DIR)

    # 1. Load & Slice
    print("Loading data...")
    base_xyz, base_opacity = load_ply_xyz_opacity(BASELINE_PLY)
    ours_xyz, ours_opacity = load_ply_xyz_opacity(OURS_PLY)
    
    base_xy, base_w = slice_points(base_xyz, base_opacity, SLICE_AXIS, SLICE_RANGE, VIEW_AXES)
    ours_xy, ours_w = slice_points(ours_xyz, ours_opacity, SLICE_AXIS, SLICE_RANGE, VIEW_AXES)

    # 2. Dynamic Aspect Ratio with Cropping
    hist_range = [CROP_Z, CROP_Y]
    
    z_span = CROP_Z[1] - CROP_Z[0]
    y_span = CROP_Y[1] - CROP_Y[0]
    
    bins_z = BINS_Z
    bins_y = int(BINS_Z * (y_span / z_span)) 
    dynamic_bins = [bins_z, bins_y]
    
    print(f"Physical Span: Z={z_span:.1f}m, Y={y_span:.1f}m")
    print(f"Bins: Z={bins_z}, Y={bins_y}")

    # 3. Compute Histograms
    print("Computing Histograms...")
    H_base, z_edges, y_edges = compute_histogram(base_xy, base_w, bins=dynamic_bins, hist_range=hist_range)
    H_ours, _, _ = compute_histogram(ours_xy, ours_w, bins=dynamic_bins, hist_range=hist_range)

    # Clean data (Mask background)
    H_base[H_base == 0] = np.nan
    H_ours[H_ours == 0] = np.nan

    # 4. Normalize (Adjust Gamma for visibility)
    # 调低 Gamma (0.5 -> 0.4) 可以让稀疏的点看起来更亮、更明显
    finite_vals = np.concatenate([H_base[np.isfinite(H_base)], H_ours[np.isfinite(H_ours)]])
    vmin = finite_vals[finite_vals > 0].min()
    vmax = np.percentile(finite_vals[finite_vals > 0], 99.5) # 稍微砍掉一点极亮值，提高整体亮度
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax) 
    extent = [z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]]

    # =========================================================================
    # 🎨 PLOTTING 
    # =========================================================================
    print("Rendering...")
    
    # 调整画布比例，使其更紧凑
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    plt.subplots_adjust(wspace=0.02, left=0.08, right=0.98, top=0.95, bottom=0.15)

    datasets = [("3DGS", H_base), ("Ours", H_ours)]

    for i, (ax, (name, hist)) in enumerate(zip(axes, datasets)):
        im = ax.imshow(
            hist.T, origin="lower", extent=extent,
            cmap=CMAP, norm=norm, interpolation="nearest", aspect='equal'
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- 修复标签遮挡问题 ---
        # 加上 bbox (半透明白底)，防止文字看不清
        txt = ax.text(0.04, 0.90, name, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color='black', ha='left', va='top')
        txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        ax.set_xlabel("Depth (Z, m)", fontsize=11, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("Height (-Y, m)", fontsize=11, fontweight='bold')
        else:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False) 
            
            # --- 悬浮 Colorbar ---
            axins = inset_axes(ax, width="4%", height="35%", loc='upper right', 
                             bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=3)
            cbar = fig.colorbar(im, cax=axins, orientation="vertical")
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels(["Sparse", "Dense"]) 
            cbar.ax.tick_params(labelsize=9, color='white', labelcolor='black', length=0)
            cbar.outline.set_edgecolor('white')
            cbar.outline.set_linewidth(1)

    save_path = os.path.join(OUTPUT_DIR, "tomography.pdf")
    print(f"Saving to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()