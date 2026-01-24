"""
Mass-Aware Densification Visualization Script for GlowGS Paper

Creates an intuitive visualization showing mass-aware weight distribution
overlaid on a rendered image. 

Key insight: We use RENDERED image + edge/gradient information as a proxy
for mass-aware weights since actual screen-space radii require full rendering pipeline.

Regions with:
- High gradient + varying opacity → HIGH weight (thin structures like spokes)
- Low gradient + high opacity → LOW weight (large solid regions like ground)

Black & white or heatmap overlay, no title, no legend - suitable for method figure.

Usage:
    python paper_experiments/07_mass_aware/visualize_mass_aware.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def compute_mass_aware_proxy(image_np):
    """
    Compute a proxy for mass-aware weight distribution from a 2D image.
    
    The key insight of mass-aware densification:
    - High weight for THIN structures (high local gradient, small "radius")
    - Low weight for LARGE solid regions (low gradient, large "radius")
    
    We approximate this using image gradients:
    - Edge regions → small Gaussians → high weight
    - Flat regions → large Gaussians → low weight (mass penalty)
    
    Args:
        image_np: [H, W, 3] RGB image in [0, 1]
    
    Returns:
        weight_map: [H, W] mass-aware weight proxy
        edge_map: [H, W] edge strength
        flatness_map: [H, W] flatness (inverse of edge)
    """
    # Convert to grayscale
    gray = 0.299 * image_np[:,:,0] + 0.587 * image_np[:,:,1] + 0.114 * image_np[:,:,2]
    
    # Compute edge strength using Sobel
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    edge_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize edge to [0, 1]
    edge_norm = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-6)
    
    # Compute local variance as another edge indicator (captures texture)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray, size=11)
    local_sqr_mean = uniform_filter(gray**2, size=11)
    local_var = np.maximum(local_sqr_mean - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    local_std_norm = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-6)
    
    # Combine edge and texture for "detail score"
    detail_score = 0.6 * edge_norm + 0.4 * local_std_norm
    
    # Mass-aware weight proxy:
    # - Detail regions (edges, texture): HIGH weight → visibility boost dominates
    # - Flat regions (ground, sky): LOW weight → mass penalty dominates
    #
    # Formula interpretation:
    # - In real mass-aware: weight = sqrt(α) / (1 + α*r/τ)
    # - Thin structures: small r → weight ≈ sqrt(α) → high
    # - Large blobs: large r → weight ≈ sqrt(α)/(α*r/τ) → low
    # - Detail score is a proxy for 1/r (inverse of Gaussian size)
    
    weight_map = detail_score ** 0.8  # Slight gamma for better contrast
    
    # Smooth slightly for cleaner visualization
    weight_map = gaussian_filter(weight_map, sigma=1)
    
    return weight_map, edge_norm, 1.0 - edge_norm


def create_heatmap_overlay(image_np, weight_map, alpha=0.6):
    """
    Create a heatmap overlay on the original image.
    
    Using Nature/Science style colormap (viridis):
    - Low weight → Dark purple/blue
    - High weight → Bright yellow/green
    """
    # Normalize weight to [0, 1]
    w_norm = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-6)
    
    # Use viridis colormap (Nature/Science standard, perceptually uniform)
    cmap = plt.cm.viridis
    heatmap = cmap(w_norm)[:, :, :3]  # [H, W, 3] RGB
    
    # Blend with original image
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap


def visualize_mass_aware(image_path, output_dir, image_size=None):
    """
    Generate Mass-Aware weight visualization overlaid on rendered image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Optionally resize for faster processing
    if image_size is not None:
        # Maintain aspect ratio
        w, h = img.size
        scale = image_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img_np.shape}")
    
    # Compute mass-aware weight proxy
    print("Computing mass-aware weight proxy...")
    weight_map, edge_map, flatness_map = compute_mass_aware_proxy(img_np)
    
    # ========================================================================
    # Save only essential visualizations for paper
    # ========================================================================
    
    # Main output: Heatmap overlay (Nature/Science viridis colormap)
    # This is the most intuitive visualization for method figure
    overlay, heatmap = create_heatmap_overlay(img_np, weight_map, alpha=0.5)
    overlay_img = (overlay * 255).astype(np.uint8)
    Image.fromarray(overlay_img).save(
        os.path.join(output_dir, 'mass_aware_overlay.png'))
    
    print(f"\nSaved: {output_dir}/mass_aware_overlay.png")
    print("\n" + "="*60)
    print("Mass-Aware Weight Visualization (viridis colormap)")
    print("="*60)
    print("YELLOW/GREEN regions: High weight (thin structures, edges)")
    print("  → Visibility boost dominates → MORE Gaussians allocated")
    print("PURPLE/BLUE regions: Low weight (large solid regions)")
    print("  → Mass penalty dominates → FEWER Gaussians allocated")


def main():
    # Use the same GT image as edge loss visualization
    # Priority: test GT > original image
    gt_path = "output/bicycle/test/ours_30000/gt/00000.png"
    fallback_path = "data/360_v2/bicycle/images/_DSC8679.JPG"
    
    if os.path.exists(gt_path):
        image_path = gt_path
    elif os.path.exists(fallback_path):
        image_path = fallback_path
    else:
        print("Error: No image found!")
        print(f"  Tried: {gt_path}")
        print(f"  Tried: {fallback_path}")
        return
    
    output_dir = "paper_experiments/07_mass_aware"
    visualize_mass_aware(image_path, output_dir, image_size=None)  # Keep original size


if __name__ == "__main__":
    main()
