"""Plotting utilities for hash-vs-hybrid ablation.

This module now includes reusable helpers for:
- loading rgb/depth/normal assets
- ROI cropping
- unified colormap application
- simple grid assembly for Stage2 figures
"""

# TODO(stage2-task3): enforce consistent typography and panel sizing across all figures

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import matplotlib
# Use non-interactive backend to avoid Qt/xcb dependency in headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image


def placeholder_figure(title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def load_rgb(image_path: Path) -> np.ndarray:
    """Load an RGB PNG to float32 array in [0,1]."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def load_depth_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "depth" in data:
        return data["depth"].astype(np.float32)
    # fallback key
    return list(data.values())[0].astype(np.float32)


def load_normal_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "normal" in data:
        arr = data["normal"].astype(np.float32)
    else:
        arr = list(data.values())[0].astype(np.float32)
    # Expect shape HxWx3 in [-1,1]; map to 0-1 for visualization
    arr = (arr * 0.5 + 0.5).clip(0.0, 1.0)
    return arr


def crop_roi(img: np.ndarray, roi: List[int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return img[y0:y1, x0:x1]


def apply_depth_colormap(depth: np.ndarray, vmin: Optional[float], vmax: Optional[float], cmap_name: str = "turbo") -> Tuple[np.ndarray, float, float]:
    """Apply colormap to depth with shared vmin/vmax. Returns rgb image and used limits."""
    if vmin is None:
        vmin = float(np.nanpercentile(depth, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(depth, 99))
    vmin = float(vmin)
    vmax = float(max(vmax, vmin + 1e-6))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colored = cmap(norm(depth))[..., :3]
    return colored, vmin, vmax


def make_grid(panels: List[np.ndarray], nrow: int, ncol: int, pad: int = 4, bg: float = 1.0) -> np.ndarray:
    """Assemble panels (HWC float) into a grid with padding."""
    assert len(panels) == nrow * ncol
    h, w, c = panels[0].shape
    out = np.full((nrow * h + pad * (nrow - 1), ncol * w + pad * (ncol - 1), c), bg, dtype=np.float32)
    idx = 0
    for r in range(nrow):
        for ccol in range(ncol):
            y0 = r * (h + pad)
            x0 = ccol * (w + pad)
            out[y0:y0 + h, x0:x0 + w] = panels[idx]
            idx += 1
    return out


def save_image(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(path)


def assemble_rgb_depth_normal(scene: str, method: str, rgb_path: Path, depth_npz: Path, normal_npz: Path, roi_list: List[List[int]], depth_range: Dict[str, Any], save_path: Path) -> Dict[str, Any]:
    """Build 2-row figure: RGB / depth / normal / two ROI crops.

    Returns a manifest dict with used ranges and inputs.
    """
    rgb = load_rgb(rgb_path)
    depth = load_depth_npz(depth_npz)
    normal = load_normal_npz(normal_npz)

    depth_rgb, vmin, vmax = apply_depth_colormap(depth, depth_range.get("min"), depth_range.get("max"))

    rois = [crop_roi(rgb, roi) for roi in roi_list]
    panels = [rgb, depth_rgb, normal] + rois

    # Resize panels to the smallest height among them for grid consistency
    h_min = min(p.shape[0] for p in panels)
    panels_resized = []
    for p in panels:
        if p.shape[0] == h_min:
            panels_resized.append(p)
        else:
            img = Image.fromarray((p * 255).astype(np.uint8))
            scale = h_min / p.shape[0]
            new_w = int(p.shape[1] * scale)
            img = img.resize((new_w, h_min), Image.BILINEAR)
            panels_resized.append(np.asarray(img).astype(np.float32) / 255.0)

    grid = make_grid(panels_resized, nrow=1, ncol=len(panels_resized), pad=6, bg=1.0)
    save_image(grid, save_path)

    return {
        "scene": scene,
        "method": method,
        "inputs": {
            "rgb": str(rgb_path),
            "depth": str(depth_npz),
            "normal": str(normal_npz),
            "rois": roi_list,
        },
        "depth_range": {"min": vmin, "max": vmax},
        "output": str(save_path),
    }


