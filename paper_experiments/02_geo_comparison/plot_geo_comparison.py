#!/usr/bin/env python3
"""
Save cropped Error maps and Depth maps (magma/black style) for two methods.

Outputs:
- <Method>_error.png           : Zoomed error crop (with colorbar)
- <Method>_depth.png           : Full depth map, magma, no colorbar
- <Method>_foliage.png         : Top-left foliage detail (depth), magma, no colorbar
- <Method>_bench_noise.png     : Bottom-center bench detail (depth), magma, no colorbar
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Input configuration (edit these paths and labels as needed)
# -----------------------------------------------------------------------------
GT_RGB_PATH: Optional[str] = "/home/ubuntu/lyj/Project/GlowGS/output/legacy_output/bicycle/test/ours_30000/gt/00000.png"  # e.g., "/path/to/gt.png"; if None, expect rgb_gt in NPZ
ERROR_CMAP = "inferno"             # Error map colormap (kept from previous style)
# Depth colormap: mirror historical Ours_Depth (magma_r with black background)
DEPTH_CMAP_NAME = "magma_r"

# Error zoom box (pixel space of full image)
ZOOM_BOX = (2200, 1450, 420, 420)   # (x, y, w, h)
ZOOM_SCALE = 4

# Detail crops (fractions of H/W to preserve original aspect)
FOLIAGE_CROP = (0.0, 0.35, 0.0, 0.35)   # y0, y1, x0, x1 (fractions)
BENCH_CROP = (0.60, 0.85, 0.30, 0.70)
DETAIL_SCALE = 2  # Upscale detail crops for visibility


@dataclass
class MethodEntry:
    name: str
    npz_path: str


METHODS: List[MethodEntry] = [
    MethodEntry(
        name="Hash Only",
        npz_path="/home/ubuntu/lyj/Project/GlowGS/bicycle_hash_only.npz",
    ),
    MethodEntry(
        name="Hash + VM",
        npz_path="/home/ubuntu/lyj/Project/GlowGS/ours_view0.npz",
    ),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_unit_range(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.max() > 1.5:
        return np.clip(arr / 255.0, 0.0, 1.0)
    return np.clip(arr, 0.0, 1.0)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    img = _to_unit_range(img)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:]
        return rgb * alpha
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError("Input must be RGB or RGBA array.")


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def load_single_npz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not path or not os.path.isfile(path) or not path.endswith(".npz"):
        raise FileNotFoundError(f"NPZ missing: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError:
        data = np.load(path, allow_pickle=True)

    rgb = ensure_rgb(data["rgb"])
    depth = np.asarray(data.get("depth"), dtype=np.float32) if "depth" in data else None
    rgb_gt = ensure_rgb(data.get("rgb_gt")) if "rgb_gt" in data else None
    return rgb, depth, rgb_gt


def mask_depth(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=np.float32)
    return np.where(depth > 0, depth, np.nan)


def load_gt_image(path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (target_shape[1], target_shape[0]):
        img = img.resize((target_shape[1], target_shape[0]), resample=Image.BICUBIC)
    return ensure_rgb(np.asarray(img))


def crop_xywh(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("Zoom box is outside the image bounds; adjust ZOOM_BOX.")
    return patch


def crop_fraction(image: np.ndarray, frac_box: Tuple[float, float, float, float]) -> np.ndarray:
    y0f, y1f, x0f, x1f = frac_box
    h, w = image.shape[:2]
    y0, y1 = int(h * y0f), int(h * y1f)
    x0, x1 = int(w * x0f), int(w * x1f)
    return image[y0:y1, x0:x1]


def upscale_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return patch
    h, w = patch.shape[:2]
    new_size = (w * scale, h * scale)
    safe = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(Image.fromarray((safe * 255).astype(np.uint8)).resize(new_size, resample=Image.BICUBIC)) / 255.0


def save_map_with_colorbar(img: np.ndarray, vmin: float, vmax: float, label: str, path: str, cmap: str) -> None:
    """Save map (error) with inset colorbar in lower-right corner."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.font_manager as fm

    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 200, h / 200), dpi=200, facecolor="black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])  # Fill entire figure

    axins = inset_axes(
        ax,
        width="5%",
        height="40%",
        loc="lower right",
        borderpad=3.0,
    )

    fig.colorbar(im, cax=axins, orientation="vertical", ticks=[vmin, vmax])
    axins.tick_params(axis="y", which="both", length=0)

    font_props = fm.FontProperties(family="DejaVu Sans", weight="bold", size=10)

    if vmax >= 1:
        labels = [f"{vmin:.0f}", f"{vmax:.0f}"]
    else:
        labels = [f"{vmin:.1f}", f"{vmax:.1f}"]
    axins.set_yticklabels(labels, fontproperties=font_props, color="white")

    axins.set_title(label, fontproperties=font_props, color="white", pad=4)

    for spine in axins.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)

    fig.savefig(path, dpi=300, pad_inches=0, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


def save_map_plain(img: np.ndarray, vmin: float, vmax: float, path: str, cmap_name: str) -> None:
    """Save map without colorbar/axes on black background."""
    h, w = img.shape[:2]
    fig_w = w / 400
    fig_h = h / 400
    cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black", dpi=200)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(path, dpi=300, pad_inches=0, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


def depth_to_rgb(depth_vis: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    """Map depth (with NaNs) to RGB using shared vmin/vmax and black for invalid."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    h, w = depth_vis.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.isfinite(depth_vis)
    if mask.any():
        rgb[mask] = cmap(norm(depth_vis[mask]))[:, :3]
    return rgb


def save_rgb_plain(rgb: np.ndarray, path: str) -> None:
    """Save RGB image (0-1) on black background without axes/colorbar."""
    h, w = rgb.shape[:2]
    fig_w = w / 400
    fig_h = h / 400
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black", dpi=200)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(rgb, vmin=0.0, vmax=1.0)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(path, dpi=300, pad_inches=0, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    records: List[Dict[str, np.ndarray]] = []
    gt_rgb: Optional[np.ndarray] = None

    for entry in METHODS:
        rgb, depth, rgb_gt = load_single_npz(entry.npz_path)
        rgb = gamma_correct(rgb)
        depth_masked = mask_depth(depth)
        records.append({
            "name": entry.name,
            "rgb": rgb,
            "depth": depth,
            "depth_vis": depth_masked,
        })
        if gt_rgb is None and rgb_gt is not None:
            gt_rgb = gamma_correct(rgb_gt)

    if gt_rgb is None:
        if GT_RGB_PATH is None:
            raise ValueError("Provide GT_RGB_PATH or include rgb_gt in NPZ.")
        gt_rgb = load_gt_image(GT_RGB_PATH, target_shape=records[0]["rgb"].shape[:2])
        gt_rgb = gamma_correct(gt_rgb)

    # Compute error maps for all methods
    error_maps = []
    for rec in records:
        error = np.mean(np.abs(rec["rgb"] - gt_rgb), axis=-1)
        error_maps.append(error)

    # Shared error range across methods
    stacked_errors = np.concatenate([e.flatten() for e in error_maps])
    vmax_err = np.percentile(stacked_errors, 99.5)
    vmax_err = max(vmax_err, 1e-4)

    # Shared depth range across methods (valid depths), match historical 2-98 percentiles
    depth_values = []
    for rec in records:
        if rec["depth_vis"] is not None:
            valid_depth = rec["depth_vis"][np.isfinite(rec["depth_vis"])]
            if valid_depth.size > 0:
                depth_values.append(valid_depth)
    if depth_values:
        all_depths = np.concatenate(depth_values)
        vmin_depth = np.percentile(all_depths, 2)
        vmax_depth = np.percentile(all_depths, 98)
        if vmin_depth == vmax_depth:
            vmax_depth = vmin_depth + 1e-6
    else:
        vmin_depth, vmax_depth = 0, 1

    # Save per-method outputs
    for rec, error in zip(records, error_maps):
        name = rec["name"]
        depth = rec["depth"]
        depth_vis = rec["depth_vis"]

        # 1) Cropped error (zoom) with colorbar
        zoom_err = crop_xywh(error, ZOOM_BOX)
        zoom_err = upscale_patch(zoom_err, ZOOM_SCALE)
        err_path = os.path.join(OUTPUT_DIR, f"{name}_error.png")
        save_map_with_colorbar(
            zoom_err,
            vmin=0.0,
            vmax=vmax_err,
            label="MAE",
            path=err_path,
            cmap=ERROR_CMAP,
        )

        # 2) Full depth map (magma, black, no colorbar)
        if depth_vis is not None:
            depth_rgb = depth_to_rgb(depth_vis, vmin_depth, vmax_depth, DEPTH_CMAP_NAME)

            depth_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}_depth.png")
            save_rgb_plain(depth_rgb, depth_path)

            # 3) Detail crops (foliage and bench), saved per method from RGB depth
            for tag, frac in [("foliage", FOLIAGE_CROP), ("bench_noise", BENCH_CROP)]:
                detail = crop_fraction(depth_rgb, frac)
                detail = upscale_patch(detail, DETAIL_SCALE)
                detail_path = os.path.join(OUTPUT_DIR, f"{name}_{tag}.png")
                save_rgb_plain(detail, detail_path)
        else:
            print(f"Warning: No depth data for {rec['name']}")

    print(f"\nAll images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()