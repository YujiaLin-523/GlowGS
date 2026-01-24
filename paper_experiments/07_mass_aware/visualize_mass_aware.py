"""
Mass-Aware Densification Visualization Script for GlowGS Paper

Visualizes the Mass-Aware Gradient Weighting mechanism on REAL Gaussian data:
- Load trained Gaussians from PLY file
- Project to 2D image space
- Color by mass-aware weight

Black & white, no title, no legend - suitable for method figure.

Usage:
    python paper_experiments/07_mass_aware/visualize_mass_aware.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image
from plyfile import PlyData
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_gaussians_from_ply(ply_path):
    """
    Load Gaussian properties from PLY file.
    
    Returns:
        xyz: [N, 3] positions
        opacity: [N] opacity values (after sigmoid)
        scales: [N, 3] scale values (after exp)
    """
    print(f"Loading Gaussians from: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    xyz = np.stack([
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ], axis=1)
    
    # Opacity is stored as inverse sigmoid, apply sigmoid
    opacity_raw = np.asarray(plydata.elements[0]["opacity"])
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    
    # Scales are stored as log, apply exp
    scales = np.stack([
        np.exp(np.asarray(plydata.elements[0]["scale_0"])),
        np.exp(np.asarray(plydata.elements[0]["scale_1"])),
        np.exp(np.asarray(plydata.elements[0]["scale_2"]))
    ], axis=1)
    
    print(f"  Loaded {len(xyz)} Gaussians")
    print(f"  Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    
    return xyz, opacity, scales


def compute_mass_aware_weight(opacity, radius, tau=100.0, mass_scale=0.1):
    """
    Compute Mass-Aware gradient weight.
    
    Formula: weight = sqrt(α) * 1/(1 + mass_scale * α * r / τ)
    """
    alpha = np.clip(opacity, 0.01, 1.0)
    r = np.clip(radius, 1.0, None)
    
    visibility_boost = np.sqrt(alpha)
    mass = alpha * r
    mass_penalty = 1.0 / (1.0 + mass_scale * mass / tau)
    
    return visibility_boost * mass_penalty


def project_gaussians_to_image(xyz, weights, image_shape, camera_center=None, scale=1.0):
    """
    Simple orthographic projection of Gaussians to 2D image.
    Uses X-Z plane for top-down view or X-Y for front view.
    
    Args:
        xyz: [N, 3] world positions
        weights: [N] per-Gaussian weights
        image_shape: (H, W) output image size
        camera_center: Optional center point for projection
        scale: Pixels per world unit
    
    Returns:
        weight_map: [H, W] accumulated weight image
    """
    H, W = image_shape
    
    if camera_center is None:
        camera_center = xyz.mean(axis=0)
    
    # Use X-Y plane (front view) - more intuitive for bicycle
    x = xyz[:, 0] - camera_center[0]
    y = xyz[:, 1] - camera_center[1]
    
    # Auto-scale to fit image
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    auto_scale = min(W / (x_range + 1e-6), H / (y_range + 1e-6)) * 0.9
    
    # Convert to pixel coordinates
    px = ((x * auto_scale) + W / 2).astype(np.int32)
    py = ((y * auto_scale) + H / 2).astype(np.int32)
    
    # Flip Y for image coordinates (top=0)
    py = H - 1 - py
    
    # Clamp to image bounds
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, w = px[valid], py[valid], weights[valid]
    
    # Accumulate weights into image
    weight_map = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)
    
    np.add.at(weight_map, (py, px), w)
    np.add.at(count_map, (py, px), 1)
    
    # Average where multiple points project to same pixel
    count_map = np.maximum(count_map, 1)
    weight_map = weight_map / count_map
    
    return weight_map


def visualize_mass_aware(ply_path, output_dir, image_size=1024):
    """
    Generate Mass-Aware weight visualization from real Gaussian data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Gaussians
    xyz, opacity, scales = load_gaussians_from_ply(ply_path)
    
    # Compute "radius" as mean scale (proxy for screen-space size)
    # In real rendering, this depends on camera distance, but mean scale is a good proxy
    mean_scale = scales.mean(axis=1)
    
    # Normalize radius to reasonable range [1, 200] for mass-aware formula
    radius = 1 + 199 * (mean_scale - mean_scale.min()) / (mean_scale.max() - mean_scale.min() + 1e-6)
    
    print(f"\nComputing Mass-Aware weights...")
    print(f"  Radius range (normalized): [{radius.min():.1f}, {radius.max():.1f}]")
    
    # Compute mass-aware weights
    mass_weights = compute_mass_aware_weight(opacity, radius, tau=100.0, mass_scale=0.1)
    
    # Also compute individual components
    visibility_boost = np.sqrt(np.clip(opacity, 0.01, 1.0))
    mass = np.clip(opacity, 0.01, 1.0) * radius
    mass_penalty = 1.0 / (1.0 + 0.1 * mass / 100.0)
    
    print(f"  Weight range: [{mass_weights.min():.3f}, {mass_weights.max():.3f}]")
    
    # Project to 2D images
    img_shape = (image_size, image_size)
    
    print(f"\nProjecting {len(xyz)} Gaussians to {image_size}x{image_size} image...")
    
    # 1. Mass-Aware Weight Map (main output)
    weight_map = project_gaussians_to_image(xyz, mass_weights, img_shape)
    
    # 2. Visibility Boost Map
    visibility_map = project_gaussians_to_image(xyz, visibility_boost, img_shape)
    
    # 3. Mass Penalty Map
    penalty_map = project_gaussians_to_image(xyz, mass_penalty, img_shape)
    
    # 4. Opacity Map (for reference)
    opacity_map = project_gaussians_to_image(xyz, opacity, img_shape)
    
    # 5. Radius Map (for reference, inverted so small=bright)
    radius_norm = 1.0 - (radius - radius.min()) / (radius.max() - radius.min() + 1e-6)
    radius_map = project_gaussians_to_image(xyz, radius_norm, img_shape)
    
    # Apply Gaussian blur to smooth the sparse projections
    from scipy.ndimage import gaussian_filter
    sigma = 3
    weight_map = gaussian_filter(weight_map, sigma=sigma)
    visibility_map = gaussian_filter(visibility_map, sigma=sigma)
    penalty_map = gaussian_filter(penalty_map, sigma=sigma)
    opacity_map = gaussian_filter(opacity_map, sigma=sigma)
    radius_map = gaussian_filter(radius_map, sigma=sigma)
    
    # Normalize and save as grayscale images
    def save_normalized(arr, filename):
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        img = (arr_norm * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(os.path.join(output_dir, filename))
    
    save_normalized(weight_map, 'mass_aware_weight.png')
    save_normalized(visibility_map, 'visibility_boost.png')
    save_normalized(penalty_map, 'mass_penalty.png')
    save_normalized(opacity_map, 'opacity.png')
    save_normalized(radius_map, 'radius_inv.png')
    
    print(f"\nSaved visualizations to: {output_dir}")
    print("Files generated:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


def main():
    # Use the trained bicycle Gaussian model
    ply_path = "output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
    output_dir = "paper_experiments/07_mass_aware"
    
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        print("Please train a model first: python train.py -s data/360_v2/bicycle -m output/bicycle --eval")
        return
    
    visualize_mass_aware(ply_path, output_dir, image_size=1024)


if __name__ == "__main__":
    main()
