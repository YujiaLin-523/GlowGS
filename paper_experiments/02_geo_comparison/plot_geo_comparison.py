#!/usr/bin/env python3
"""
RGB + zoom + error + normal evidence chain for 3DGS vs GlowGS.
This script saves every required subplot individually (no need to stitch).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Input configuration (edit these paths and labels as needed)
# -----------------------------------------------------------------------------
GT_RGB_PATH: Optional[str] = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/test/ours_30000/gt/00000.png"  # e.g., "/path/to/gt.png"; if None, expect rgb_gt in NPZ
ZOOM_BOX = (2200, 1450, 420, 420)  # (x, y, w, h) in pixel space of the full image
ZOOM_SCALE = 4                     # ×4 or ×8 magnification label appears on zoomed patches
RECT_COLOR = "#00A0FF"            # uniform zoom box color
RECT_LINEWIDTH = 2.5
ERROR_CMAP = "inferno"             # high contrast: black-red-yellow
FIG_BG = "#FFFFFF"


@dataclass
class MethodEntry:
    name: str
    npz_path: str
    size_fps: str  # text shown on the RGB corner, e.g., "49 MB / 65 FPS"


METHODS: List[MethodEntry] = [
    MethodEntry(
        name="3DGS",
        npz_path="/home/ubuntu/lyj/Project/gaussian-splatting/baseline_view0.npz",
        size_fps="734 MB / 134 FPS",
    ),
    MethodEntry(
        name="GlowGS",
        npz_path="/home/ubuntu/lyj/Project/GlowGS/ours_view0.npz",
        size_fps="11 MB / 195 FPS",
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


def prepare_normal(norm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if norm is None:
        return None
    mapped = (np.asarray(norm, dtype=np.float32) + 1.0) / 2.0
    mask_black = np.all(norm == 0.0, axis=-1, keepdims=True)
    return np.where(mask_black, 0.0, mapped)


def load_single_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not path or not os.path.isfile(path) or not path.endswith(".npz"):
        raise FileNotFoundError(f"NPZ missing: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError:
        data = np.load(path, allow_pickle=True)

    rgb = ensure_rgb(data["rgb"])
    depth = np.asarray(data.get("depth"), dtype=np.float32) if "depth" in data else None
    norm = np.asarray(data.get("norm"), dtype=np.float32) if "norm" in data else None
    rgb_gt = ensure_rgb(data.get("rgb_gt")) if "rgb_gt" in data else None
    return rgb, depth, norm, rgb_gt


def load_gt_image(path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (target_shape[1], target_shape[0]):
        img = img.resize((target_shape[1], target_shape[0]), resample=Image.BICUBIC)
    return ensure_rgb(np.asarray(img))


def crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("Zoom box is outside the image bounds; adjust ZOOM_BOX.")
    return patch


def upscale_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    h, w = patch.shape[:2]
    new_size = (w * scale, h * scale)
    return np.asarray(Image.fromarray((patch * 255).astype(np.uint8)).resize(new_size, resample=Image.BICUBIC)) / 255.0


def draw_box_on_rgb(rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    img = (rgb * 255).clip(0, 255).astype(np.uint8).copy()
    x, y, w, h = box
    t = max(1, int(round(RECT_LINEWIDTH)))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    # Top and bottom edges
    img[y0:y0 + t, x0:x1, :] = np.array(ImageColor.getrgb(RECT_COLOR), dtype=np.uint8)
    img[y1 - t:y1, x0:x1, :] = np.array(ImageColor.getrgb(RECT_COLOR), dtype=np.uint8)
    # Left and right edges
    img[y0:y1, x0:x0 + t, :] = np.array(ImageColor.getrgb(RECT_COLOR), dtype=np.uint8)
    img[y0:y1, x1 - t:x1, :] = np.array(ImageColor.getrgb(RECT_COLOR), dtype=np.uint8)
    return img.astype(np.float32) / 255.0


def annotate_rgb(rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, mode="RGB")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    text_pos = (int(0.02 * img.width), int(0.05 * img.height))
    padding = 4
    bbox = draw.textbbox(text_pos, text, font=font)
    bg_rect = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(bg_rect, fill=(0, 0, 0))
    draw.text(text_pos, text, fill=(255, 255, 255), font=font)
    return np.asarray(img).astype(np.float32) / 255.0


def save_with_colorbar(img: np.ndarray, vmin: int, vmax: int, label: str, path: str, cmap) -> None:
    """Save error map with inset colorbar in lower-right corner."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.font_manager as fm
    
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 200, h / 200), dpi=200)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])  # Fill entire figure
    
    # Inset colorbar in lower-right corner (vertical, 40% height)
    axins = inset_axes(
        ax,
        width="5%",
        height="40%",
        loc="lower right",
        borderpad=3.0,  # Increased to shift colorbar left
    )
    
    cbar = fig.colorbar(im, cax=axins, orientation="vertical", ticks=[vmin, vmax])
    
    # Remove tick marks (length=0)
    axins.tick_params(axis="y", which="both", length=0)
    
    # Font: DejaVu Sans bold white (cross-platform compatible)
    font_props = fm.FontProperties(family="DejaVu Sans", weight="bold", size=10)
    
    # Set tick labels
    axins.set_yticklabels(["0", "1"], fontproperties=font_props, color="white")
    
    # Title (MAE) above colorbar
    axins.set_title("MAE", fontproperties=font_props, color="white", pad=4)
    
    # Thin white border around colorbar
    for spine in axins.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)
    
    fig.savefig(path, dpi=300, pad_inches=0)
    plt.close(fig)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    records: List[Dict[str, np.ndarray]] = []
    gt_rgb: Optional[np.ndarray] = None

    for entry in METHODS:
        rgb, depth, norm, rgb_gt = load_single_npz(entry.npz_path)
        rgb = gamma_correct(rgb)
        norm = prepare_normal(norm)
        records.append({
            "name": entry.name,
            "size_fps": entry.size_fps,
            "rgb": rgb,
            "depth": depth,
            "norm": norm,
        })
        if gt_rgb is None and rgb_gt is not None:
            gt_rgb = gamma_correct(rgb_gt)

    if gt_rgb is None:
        if GT_RGB_PATH is None:
            raise ValueError("Provide GT_RGB_PATH or include rgb_gt in NPZ.")
        gt_rgb = load_gt_image(GT_RGB_PATH, target_shape=records[0]["rgb"].shape[:2])
        gt_rgb = gamma_correct(gt_rgb)

    # Shared error range across methods
    error_maps = []
    for rec in records:
        error = np.mean(np.abs(rec["rgb"] - gt_rgb), axis=-1)
        error_maps.append(error)
    stacked_errors = np.concatenate([e.flatten() for e in error_maps])
    vmax_err = np.percentile(stacked_errors, 99.5)
    vmax_err = max(vmax_err, 1e-4)

    # Save per-method subplots
    for rec, error in zip(records, error_maps):
        name = rec["name"]
        rgb = rec["rgb"]
        norm = rec["norm"]

        boxed = draw_box_on_rgb(rgb, ZOOM_BOX)
        boxed = annotate_rgb(boxed, rec["size_fps"])
        # plt.imsave(os.path.join(OUTPUT_DIR, f"{name}_rgb_full.png"), boxed)

        zoom_rgb = crop(rgb, ZOOM_BOX)
        zoom_rgb = upscale_patch(zoom_rgb, ZOOM_SCALE)
        zoom_fig, zoom_ax = plt.subplots(figsize=(zoom_rgb.shape[1] / 200, zoom_rgb.shape[0] / 200), dpi=200)
        zoom_ax.imshow(zoom_rgb)
        zoom_ax.text(0.02, 0.95, f"×{ZOOM_SCALE}", color="white", fontsize=12, fontweight="bold", transform=zoom_ax.transAxes, bbox=dict(facecolor="black", alpha=0.65, pad=3, edgecolor="none"), va="top")
        zoom_ax.axis("off")
        zoom_fig.tight_layout(pad=0)
        # zoom_fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_zoom_rgb.png"), bbox_inches="tight", dpi=300, facecolor=FIG_BG)
        plt.close(zoom_fig)

        zoom_err = crop(error, ZOOM_BOX)
        zoom_err = upscale_patch(zoom_err, ZOOM_SCALE)
        save_with_colorbar(zoom_err, vmin=0, vmax=int(vmax_err), label="Error", path=os.path.join(OUTPUT_DIR, f"{name}_error.png"), cmap=ERROR_CMAP)

        if norm is None:
            raise ValueError(f"Missing normals for {name}; required for geometry evidence.")
        norm_patch = crop(norm, ZOOM_BOX)
        norm_patch = upscale_patch(norm_patch, ZOOM_SCALE)
        norm_fig, norm_ax = plt.subplots(figsize=(norm_patch.shape[1] / 200, norm_patch.shape[0] / 200), dpi=200)
        norm_ax.imshow(norm_patch)
        norm_ax.axis("off")
        norm_ax.set_title("Normal")
        norm_fig.tight_layout(pad=0)
        # norm_fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_normal.png"), bbox_inches="tight", dpi=300, facecolor=FIG_BG)
        plt.close(norm_fig)

    # Optional combined grid for quick sanity check
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), facecolor=FIG_BG)
    col_titles = ["RGB", "Zoom", "", "Normal"]
    for c, title in enumerate(col_titles):
        if title:
            axes[0, c].set_title(title)

    for r, rec in enumerate(records):
        axes[r, 0].imshow(draw_box_on_rgb(rec["rgb"], ZOOM_BOX))
        axes[r, 0].axis("off")
        axes[r, 0].text(-0.06, 0.5, rec["name"], transform=axes[r, 0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")

        axes[r, 1].imshow(upscale_patch(crop(rec["rgb"], ZOOM_BOX), ZOOM_SCALE))
        axes[r, 1].axis("off")

        axes[r, 2].imshow(upscale_patch(crop(error_maps[r], ZOOM_BOX), ZOOM_SCALE), vmin=0.0, vmax=vmax_err, cmap=ERROR_CMAP)
        axes[r, 2].axis("off")

        axes[r, 3].imshow(upscale_patch(crop(records[r]["norm"], ZOOM_BOX), ZOOM_SCALE))
        axes[r, 3].axis("off")

    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)
    fig.savefig(os.path.join(OUTPUT_DIR, "figure_2_geo_quickgrid.png"), dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


if __name__ == "__main__":
    main()