"""
Edge Loss Visualization Script for GlowGS Paper

Generates edge confidence map for method figure.
Black & white, no title, no legend.

Usage:
    python paper_experiments/visualize_edge_loss.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server


def get_sobel_kernels(device, dtype):
    """Return 3x3 Sobel kernels (X and Y)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                      dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                      dtype=dtype, device=device).view(1, 1, 3, 3)
    return kx, ky


def rgb_to_grayscale(img):
    """Convert RGB image to grayscale using BT.601 luminance weights."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
    gray = (img * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return gray


def compute_sobel_magnitude(gray_img, kx, ky):
    """Compute Sobel magnitude from grayscale image."""
    gx = F.conv2d(gray_img, kx, padding=1)
    gy = F.conv2d(gray_img, ky, padding=1)
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return magnitude


def compute_edge_confidence(gt_rgb):
    """Compute edge confidence map from GT image."""
    device = gt_rgb.device
    dtype = gt_rgb.dtype
    
    if gt_rgb.dim() == 3:
        gt_rgb = gt_rgb.unsqueeze(0)
    
    kx, ky = get_sobel_kernels(device, dtype)
    gt_gray = rgb_to_grayscale(gt_rgb)
    G_gt = compute_sobel_magnitude(gt_gray, kx, ky)
    
    # Normalize
    grad_normalizer = 4.0
    G_gt_norm = G_gt / grad_normalizer
    
    # Build edge confidence (same as in gradient_loss)
    eps = 1e-6
    G_gt_detached = G_gt_norm.detach()
    g_p95 = torch.quantile(G_gt_detached[..., ::4, ::4], 0.95).clamp(min=eps)
    grad_scale = g_p95 / 0.7 + eps
    edge_confidence = torch.clamp(G_gt_detached / grad_scale, 0.0, 1.0)
    
    return edge_confidence.squeeze()


def visualize_edge_loss(image_path, output_dir):
    """Generate edge confidence visualization (black & white, no title/legend)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img_np.shape}")
    
    # Compute edge confidence
    edge_confidence = compute_edge_confidence(img_tensor)
    edge_confidence_np = edge_confidence.cpu().numpy()
    
    # Save edge confidence as grayscale image (no matplotlib, direct PIL)
    edge_img = (edge_confidence_np * 255).astype(np.uint8)
    Image.fromarray(edge_img, mode='L').save(
        os.path.join(output_dir, 'edge_confidence.png'))
    
    print(f"\nSaved: {output_dir}/edge_confidence.png")


def main():
    image_path = "data/360_v2/bicycle/images/_DSC8679.JPG"
    output_dir = "paper_experiments/06_edge_loss"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    visualize_edge_loss(image_path, output_dir)


if __name__ == "__main__":
    main()
