#!/usr/bin/env python3
"""
Hybrid background comparison figure (white canvas, black panels) for 3DGS vs Ours.
Gamma-corrected RGB, percentile-stretched magma depth (shared vmin/vmax), normals mapped to [0,1].
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figure_2_geo_comparison.png")
# Set these to your data paths. Preferred: separate npz for baseline and ours.
NPZ_3DGS_PATH = "/home/ubuntu/lyj/Project/gaussian-splatting/baseline_view0.npz"  # e.g., "/path/to/3dgs_view0.npz"
NPZ_OURS_PATH = "/home/ubuntu/lyj/Project/GlowGS/ours_view0.npz"  # e.g., "/path/to/ours_view0.npz"
# If you instead have a single NPZ containing both methods, set COMBINED_NPZ_PATH
COMBINED_NPZ_PATH = None
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
})

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def ensure_rgb(img):
    """
    Handle RGB or RGBA. If RGBA, composite on black (to preserve dark background).
    If RGB, return as-is. Assumes values in [0,1] or [0,255].
    """
    img = _to_unit_range(img)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:]
        return rgb * alpha  # black background
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError("Input must be RGB or RGBA array.")


def gamma_correct(img, gamma=2.2):
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def prepare_depth(depth):
    """Mask invalid (<=0) as NaN for black background via set_bad."""
    depth = np.asarray(depth, dtype=np.float32)
    return np.where(depth <= 0, np.nan, depth)


def prepare_normal(norm):
    """Map normals from [-1,1] to [0,1]; keep background black (all zeros)."""
    if norm is None:
        return None
    norm = np.asarray(norm, dtype=np.float32)
    mapped = (norm + 1.0) / 2.0
    mask_black = np.all(norm == 0.0, axis=-1, keepdims=True)
    return np.where(mask_black, 0.0, mapped)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_unit_range(arr):
    """If data looks like 0-255, scale to 0-1; otherwise clip to [0,1]."""
    arr = np.asarray(arr)
    if arr.max() > 1.5:
        return np.clip(arr / 255.0, 0.0, 1.0)
    return np.clip(arr, 0.0, 1.0)


def load_from_npz(path: str):
    """
    Expect keys: rgb_3dgs, depth_3dgs, norm_3dgs, rgb_ours, depth_ours, norm_ours.
    RGB may be uint8 or float; depths/normals should be float.
    """
    if not path or not os.path.isfile(path) or not path.endswith(".npz"):
        raise FileNotFoundError(f"NPZ file missing or not a .npz: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError:
        data = np.load(path, allow_pickle=True)
    required = [
        "rgb_3dgs", "depth_3dgs", "norm_3dgs",
        "rgb_ours", "depth_ours", "norm_ours",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {missing}")
    rgb_3dgs = _to_unit_range(data["rgb_3dgs"])
    rgb_ours = _to_unit_range(data["rgb_ours"])
    depth_3dgs = np.asarray(data["depth_3dgs"], dtype=np.float32)
    depth_ours = np.asarray(data["depth_ours"], dtype=np.float32)
    norm_3dgs = np.asarray(data["norm_3dgs"], dtype=np.float32)
    norm_ours = np.asarray(data["norm_ours"], dtype=np.float32)
    return rgb_3dgs, depth_3dgs, norm_3dgs, rgb_ours, depth_ours, norm_ours


def load_single_npz(path: str):
    """
    Expect keys: rgb, depth, norm (aliases also accepted: normal).
    """
    if not path or not os.path.isfile(path) or not path.endswith(".npz"):
        raise FileNotFoundError(f"NPZ file missing or not a .npz: {path}")
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError:
        data = np.load(path, allow_pickle=True)
    rgb_key = "rgb"
    depth_key = "depth"
    norm_key = None
    if "norm" in data:
        norm_key = "norm"
    elif "normal" in data:
        norm_key = "normal"
    elif "normals" in data:
        norm_key = "normals"
    for k in [rgb_key, depth_key]:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    if norm_key is None:
        norm = None
    else:
        norm = np.asarray(data[norm_key], dtype=np.float32)
    rgb = _to_unit_range(data[rgb_key])
    depth = np.asarray(data[depth_key], dtype=np.float32)
    return rgb, depth, norm


def _valid_npz(path: str) -> bool:
    return isinstance(path, str) and path.endswith(".npz") and os.path.isfile(path)


# -----------------------------------------------------------------------------
# Data source: prefer separate NPZs; fallback to combined; else synthetic demo
# -----------------------------------------------------------------------------
if _valid_npz(NPZ_3DGS_PATH) and _valid_npz(NPZ_OURS_PATH):
    rgb_3dgs, depth_3dgs, norm_3dgs = load_single_npz(NPZ_3DGS_PATH)
    rgb_ours, depth_ours, norm_ours = load_single_npz(NPZ_OURS_PATH)
    print(f"Loaded separate NPZs:\n  3DGS: {NPZ_3DGS_PATH}\n  Ours: {NPZ_OURS_PATH}")
elif _valid_npz(COMBINED_NPZ_PATH):
    (rgb_3dgs, depth_3dgs, norm_3dgs,
     rgb_ours, depth_ours, norm_ours) = load_from_npz(COMBINED_NPZ_PATH)
    print(f"Loaded combined NPZ: {COMBINED_NPZ_PATH}")
else:
    print("No valid NPZ paths provided; using synthetic demo data.")
    H, W = 320, 480
    y, x = np.mgrid[0:H, 0:W]
    center = np.array([H / 2, W / 2])[:, None, None]
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    vignette = np.clip(1 - dist / (0.8 * max(H, W)), 0, 1)

    rgb_3dgs = np.stack([0.8 * vignette, 0.5 * vignette, 0.3 * vignette], axis=-1)
    rgb_ours = np.stack([0.7 * vignette, 0.8 * vignette, 0.4 * vignette], axis=-1)
    rgb_3dgs[dist > 0.7 * max(H, W)] = 0  # black background
    rgb_ours[dist > 0.7 * max(H, W)] = 0   # black background

    depth_3dgs = np.linspace(0, 10, W)[None, :].repeat(H, axis=0)
    depth_ours = np.linspace(2, 8, W)[None, :].repeat(H, axis=0)
    depth_3dgs[dist > 0.7 * max(H, W)] = 0
    depth_ours[dist > 0.7 * max(H, W)] = 0

    nx = (x - W / 2) / (W / 2)
    ny = (y - H / 2) / (H / 2)
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
    norm_3dgs = np.stack([nx, ny, nz], axis=-1)
    norm_ours = np.stack([nx, ny * 0.5, nz], axis=-1)
    norm_3dgs[dist > 0.7 * max(H, W)] = 0
    norm_ours[dist > 0.7 * max(H, W)] = 0

# -----------------------------------------------------------------------------
# Preprocess images
# -----------------------------------------------------------------------------
rgb_3dgs_vis = gamma_correct(ensure_rgb(rgb_3dgs))
rgb_ours_vis = gamma_correct(ensure_rgb(rgb_ours))

depth_3dgs_vis = prepare_depth(depth_3dgs)
depth_ours_vis = prepare_depth(depth_ours)

norm_3dgs_vis = prepare_normal(norm_3dgs)
norm_ours_vis = prepare_normal(norm_ours)

# Shared depth vmin/vmax using 2nd-98th percentiles of valid pixels across both
valid_depths = np.concatenate([
    depth_3dgs[np.isfinite(depth_3dgs_vis) & (depth_3dgs > 0)],
    depth_ours[np.isfinite(depth_ours_vis) & (depth_ours > 0)],
])
if valid_depths.size == 0:
    vmin, vmax = 0.0, 1.0
else:
    vmin = np.percentile(valid_depths, 2)
    vmax = np.percentile(valid_depths, 98)
    if vmin == vmax:
        vmax = vmin + 1e-6

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
ensure_output_dir(OUTPUT_DIR)
fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 7),
    gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    facecolor="white",
)
fig.patch.set_facecolor("white")

titles = ["RGB", "Depth", "Normal"]
row_labels = ["3DGS", "Ours"]

# Modified: Use magma_r (reversed) so low depth (close) is bright, high depth (far) is dark.
# This ensures objects pop against the black background.
depth_cmap = plt.cm.magma_r.copy()
depth_cmap.set_bad(color="black")

for r, (rgb_img, depth_img, norm_img) in enumerate([
    (rgb_3dgs_vis, depth_3dgs_vis, norm_3dgs_vis),
    (rgb_ours_vis, depth_ours_vis, norm_ours_vis),
]):
    for c, ax in enumerate(axes[r]):
        # 1. Main Figure Plotting
        ax.set_facecolor("black")
        if c == 0:
            ax.imshow(rgb_img)
        elif c == 1:
            ax.imshow(depth_img, cmap=depth_cmap, vmin=vmin, vmax=vmax)
        else:
            if norm_img is not None:
                ax.imshow(norm_img)
            else:
                # No normals: show black placeholder to blend with panel
                ax.imshow(np.zeros((*depth_img.shape, 3)))
        ax.axis("off")
        if r == 0:
            ax.set_title(titles[c], fontsize=14, fontweight="bold", color="#000000")
        if c == 0:
            ax.text(
                -0.08, 0.5, row_labels[r],
                fontsize=14, fontweight="bold", color="#000000",
                va="center", ha="right", transform=ax.transAxes,
            )

        # ---------------------------------------------------------------------
        # 2. Save Individual Subplots (True Original Resolution, No Crop)
        # ---------------------------------------------------------------------
        method_name = row_labels[r]  # "3DGS" or "Ours"
        col_name = titles[c]         # "RGB", "Depth", "Normal"
        sub_filename = f"{method_name}_{col_name}.png"
        sub_path = os.path.join(OUTPUT_DIR, sub_filename)

        if c == 0:  # RGB
            # Assuming rgb_img is already [0,1] float or [0,255] int
            plt.imsave(sub_path, rgb_img)
            
        elif c == 1:  # Depth
            # Use the same colormap and vmin/vmax as the main figure
            # plt.imsave handles NaNs by using the cmap's 'bad' color (black here)
            plt.imsave(sub_path, depth_img, cmap=depth_cmap, vmin=vmin, vmax=vmax)
            
        else:  # Normal
            if norm_img is not None:
                plt.imsave(sub_path, norm_img)
            else:
                # Save a purely black image of the correct size
                h, w = depth_img.shape[:2]
                black_img = np.zeros((h, w, 3), dtype=np.uint8)
                plt.imsave(sub_path, black_img)
        
        print(f"Saved individual image: {sub_path}")

# Save combined figure
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved combined figure to {OUTPUT_PATH}")