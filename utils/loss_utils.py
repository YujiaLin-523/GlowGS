#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

# -----------------------------------------------------------------------------
# SSIM Window Cache (avoid repeated allocation per iteration)
# -----------------------------------------------------------------------------
_ssim_window_cache = {}

def create_window(window_size, channel):
    """Create Gaussian window for SSIM computation with caching."""
    global _ssim_window_cache
    cache_key = (window_size, channel)
    if cache_key in _ssim_window_cache:
        return _ssim_window_cache[cache_key]
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    _ssim_window_cache[cache_key] = window
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# -----------------------------------------------------------------------------
# Sobel Kernels (cached globally)
# -----------------------------------------------------------------------------
_sobel_kernel_x = None
_sobel_kernel_y = None

# -----------------------------------------------------------------------------
# GT Gradient Cache (avoid redundant Sobel on static GT images)
# Key: (image_id, H, W) tuple to uniquely identify each GT image
# Value: precomputed Sobel gradients [2, H, W] tensor on GPU
# -----------------------------------------------------------------------------
_gt_gradient_cache = {}

def _get_sobel_kernels(device, dtype):
    """
    Return cached 3x3 Sobel kernels (X and Y).
    """
    global _sobel_kernel_x, _sobel_kernel_y
    if _sobel_kernel_x is None or _sobel_kernel_x.device != device or _sobel_kernel_x.dtype != dtype:
        # Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=dtype, device=device).view(1, 1, 3, 3)
        # Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=dtype, device=device).view(1, 1, 3, 3)
        _sobel_kernel_x = kx
        _sobel_kernel_y = ky
    return _sobel_kernel_x, _sobel_kernel_y


def _rgb_to_grayscale(img):
    """
    Convert RGB image to grayscale using BT.601 luminance weights.
    Input: [B, 3, H, W] or [3, H, W]
    Output: [B, 1, H, W]
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    # BT.601: Y = 0.299*R + 0.587*G + 0.114*B
    weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
    gray = (img * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return gray


def _compute_sobel_magnitude(gray_img, kx, ky):
    """
    Compute Sobel magnitude from grayscale image.
    Input: [B, 1, H, W] grayscale image
    Output: [B, 1, H, W] Sobel magnitude
    """
    gx = F.conv2d(gray_img, kx, padding=1)
    gy = F.conv2d(gray_img, ky, padding=1)
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return magnitude


def _compute_sobel_gradients(gray_img, kx, ky):
    """Return stacked Sobel gradients [B, 2, H, W] (gx, gy)."""
    gx = F.conv2d(gray_img, kx, padding=1)
    gy = F.conv2d(gray_img, ky, padding=1)
    return torch.cat((gx, gy), dim=1)


def _get_or_compute_gt_gradients(gt_rgb, image_id=None):
    """
    Get cached GT gradients or compute and cache them.
    
    Args:
        gt_rgb: Ground truth image [C, H, W] or [B, C, H, W]
        image_id: Optional unique identifier (e.g., camera uid)
    
    Returns:
        gt_grad: [B, 2, H, W] Sobel gradients (gx, gy)
    """
    global _gt_gradient_cache
    
    # Ensure 4D: [B, C, H, W]
    if gt_rgb.dim() == 3:
        gt_rgb = gt_rgb.unsqueeze(0)
    
    B, C, H, W = gt_rgb.shape
    
    # Create cache key: use image_id if provided, else tensor data pointer
    if image_id is not None:
        cache_key = (image_id, H, W)
    else:
        # Fallback: use tensor data pointer (works if same tensor reused)
        cache_key = (gt_rgb.data_ptr(), H, W)
    
    # Check cache
    if cache_key in _gt_gradient_cache:
        cached_grad = _gt_gradient_cache[cache_key]
        # Verify device match (in case of device migration)
        if cached_grad.device == gt_rgb.device:
            return cached_grad
        else:
            # Re-cache on new device
            _gt_gradient_cache[cache_key] = cached_grad.to(gt_rgb.device)
            return _gt_gradient_cache[cache_key]
    
    # Cache miss: compute gradients
    kx, ky = _get_sobel_kernels(gt_rgb.device, gt_rgb.dtype)
    gt_gray = _rgb_to_grayscale(gt_rgb)
    gt_grad = _compute_sobel_gradients(gt_gray, kx, ky)
    
    # Store in cache (detach to avoid retaining graph)
    _gt_gradient_cache[cache_key] = gt_grad.detach()
    
    return gt_grad


def compute_pixel_importance(gt_rgb, pred_rgb, power_edge=1.2, power_error=1.0):
    """
    Compute per-pixel importance for feature-weighted densification (GlowGS innovation #3).
    
    Design Philosophy:
    - Identifies regions requiring more Gaussian capacity: high-frequency details + hard-to-fit areas.
    - Combines two complementary signals:
      1. Edge strength (from GT Laplacian): captures intrinsic scene complexity (textures, edges).
      2. Error strength (from residual): captures current reconstruction difficulty.
    - Weighted sum (0.7 edge + 0.3 error) ensures edges are always prioritized, even when
      temporarily well-fitted, preventing premature under-representation.
    
    Usage: Per-pixel importance is aggregated to per-Gaussian importance scores via EMA,
           which then modulate densification thresholds (high importance → easier to split/clone).
    
    Args:
        gt_rgb: Ground truth image [C, H, W] or [B, C, H, W]
        pred_rgb: Predicted image [C, H, W] or [B, C, H, W]
        power_edge: Exponent for edge_strength (default 1.2)
        power_error: Exponent for error_strength (default 1.0)
    
    Returns:
        pixel_importance: [H, W] tensor with values in [0, 1]
    """
    # Ensure 4D: [B, C, H, W]
    if gt_rgb.dim() == 3:
        gt_rgb = gt_rgb.unsqueeze(0)
    if pred_rgb.dim() == 3:
        pred_rgb = pred_rgb.unsqueeze(0)
    
    # ---- Edge strength from GT Sobel ----
    # OPTIMIZATION: Reuse cached GT gradients instead of recomputing
    gt_grad = _get_or_compute_gt_gradients(gt_rgb, image_id=None)  # [B, 2, H, W]
    # Compute magnitude from cached gradients
    G_gt = torch.sqrt(gt_grad[:, 0:1, :, :] ** 2 + gt_grad[:, 1:2, :, :] ** 2 + 1e-6)  # [B, 1, H, W]
    
    # Robust percentile-based normalization (more stable than mean+std for skewed distributions)
    g_p10 = torch.quantile(G_gt, 0.10).clamp(min=1e-6)
    g_p90 = torch.quantile(G_gt, 0.90).clamp(min=1e-6)
    g_range = (g_p90 - g_p10).clamp(min=1e-6)
    # Map [p10, p90] to ~[0.1, 0.9], with saturation outside
    edge_strength = ((G_gt - g_p10) / g_range * 0.8 + 0.1).clamp(0.0, 1.0)
    edge_strength = edge_strength.squeeze(0).squeeze(0)  # [H, W]
    
    # ---- Error strength from residual ----
    residual = (pred_rgb - gt_rgb).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
    r_p10 = torch.quantile(residual, 0.10).clamp(min=1e-6)
    r_p90 = torch.quantile(residual, 0.90).clamp(min=1e-6)
    
    # Prevents noise amplification in converged scenes:
    # If error spread is small (e.g. <0.05), we shouldn't map small errors to 1.0 importance.
    r_range = (r_p90 - r_p10).clamp(min=0.05)
    
    error_strength = ((residual - r_p10) / r_range * 0.8 + 0.1).clamp(0.0, 1.0)
    error_strength = error_strength.squeeze(0).squeeze(0)  # [H, W]
    
    # ---- Combine: multiplicative gating (error-driven) ----
    # Key insight: importance should decay when error is low, even at edges.
    # Old (additive): importance = 0.7*edge + 0.3*error  → edges always high
    # New (gated): importance = error * (1 + edge)  → converges to 0 when error→0
    edge_term = edge_strength ** power_edge
    error_term = error_strength ** power_error
    pixel_importance = error_term * (1.0 + edge_term)
    
    # Normalize to [0, 1] range (max possible is 2.0 when both terms are 1.0)
    pixel_importance = (pixel_importance / 2.0).clamp(0.0, 1.0)
    
    return pixel_importance.detach()  # Detach to avoid graph retention


def compute_edge_loss(pred_rgb, gt_rgb=None, gt_grad_cached=None, gt_image_id=None, mode: str = "sobel_weighted", 
                      lambda_edge: float = 0.05, flat_weight: float = 0.5, return_components: bool = False):
    """
    Edge loss interface for ablation studies.
    
    Supports:
    - "none": No edge supervision (returns zero, no computation)
    - "sobel_weighted": Adaptive edge-aware loss with flat regularization (legacy/default)
    - "cosine_edge": Orientation-only Sobel loss (new default) using cosine similarity
    
    Args:
        pred_rgb: Predicted image [3, H, W] or [B, 3, H, W]
        gt_rgb: Ground truth image (optional if gt_grad_cached provided)
        gt_grad_cached: Precomputed GT Sobel gradients [2, H, W] - PERFORMANCE BOOST
        gt_image_id: Unique identifier for automatic GT gradient caching (e.g., camera uid)
        mode: Edge loss variant ("none", "sobel_weighted", or "cosine_edge")
        lambda_edge: Loss weight (applied internally)
        flat_weight: Weight for flat region term (used in sobel_weighted)
        return_components: Whether to return (loss, edge_term, flat_term)
    
    Returns:
        Weighted loss (lambda_edge * base_loss) or tuple if return_components=True
    """
    if mode == "none":
        # No edge loss: return zero tensor with no grad
        zero_loss = torch.tensor(0.0, device=pred_rgb.device, requires_grad=False)
        if return_components:
            return zero_loss, zero_loss, zero_loss
        return zero_loss
    
    elif mode == "cosine_edge":
        loss, edge_term, flat_term = hybrid_edge_loss(
            pred_rgb, gt_rgb=gt_rgb, gt_grad_cached=gt_grad_cached, gt_image_id=gt_image_id,
            flat_weight=flat_weight, return_components=True
        )
        if return_components:
            return lambda_edge * loss, edge_term, flat_term
        return lambda_edge * loss

    elif mode == "sobel_weighted" or mode == "laplacian_weighted":
        # GlowGS unified edge-aware loss (Now using Sobel internally)
        loss = gradient_loss(
            pred_rgb, gt_rgb=gt_rgb, gt_grad_cached=gt_grad_cached, gt_image_id=gt_image_id,
            valid_mask=None, flat_weight=flat_weight, return_components=return_components
        )
        if return_components:
            base_loss, edge_term, flat_term = loss
            return lambda_edge * base_loss, edge_term, flat_term
        return lambda_edge * loss
    
    else:
        raise ValueError(
            f"Unknown edge_loss_mode: {mode}. "
            f"Expected 'none', 'sobel_weighted', or 'cosine_edge'"
        )


def hybrid_edge_loss(pred_rgb, gt_rgb=None, gt_grad_cached=None, gt_image_id=None, flat_weight: float = 0.5, eps: float = 1e-6, return_components: bool = False):
    """
    Hybrid edge loss: cosine alignment on edge mask M, L1 magnitude suppression on flat regions.

    L_edge = M * (1 - cos) + lambda_flat * (1 - M) * ||∇I_pred||_1
    
    Args:
        pred_rgb: Predicted RGB image [3, H, W] or [B, 3, H, W]
        gt_rgb: Ground truth RGB (optional if gt_grad_cached provided)
        gt_grad_cached: Precomputed GT gradients [2, H, W] (gx, gy) - PERFORMANCE OPTIMIZATION
        gt_image_id: Unique identifier for automatic caching (e.g., camera uid)
        flat_weight: Weight for flat region regularization
        eps: Numerical stability epsilon
        return_components: Return (loss, edge_term, flat_term) if True
    """
    # Ensure 4D tensors
    if pred_rgb.dim() == 3:
        pred_rgb = pred_rgb.unsqueeze(0)

    kx, ky = _get_sobel_kernels(pred_rgb.device, pred_rgb.dtype)
    pred_gray = _rgb_to_grayscale(pred_rgb)
    pred_grad = _compute_sobel_gradients(pred_gray, kx, ky)  # [B, 2, H, W]
    
    # Use cached GT gradients if available (50% speedup)
    if gt_grad_cached is not None:
        # Explicitly provided cached gradients (highest priority)
        # gt_grad_cached is [2, H, W], expand to [1, 2, H, W]
        gt_grad = gt_grad_cached.unsqueeze(0)
    elif gt_image_id is not None and gt_rgb is not None:
        # Automatic caching using image_id (e.g., camera uid)
        gt_grad = _get_or_compute_gt_gradients(gt_rgb, image_id=gt_image_id)
    else:
        # Fallback: compute on-the-fly (backward compatibility)
        if gt_rgb is None:
            raise ValueError("Either gt_rgb or gt_grad_cached must be provided")
        if gt_rgb.dim() == 3:
            gt_rgb = gt_rgb.unsqueeze(0)
        gt_gray = _rgb_to_grayscale(gt_rgb)
        gt_grad = _compute_sobel_gradients(gt_gray, kx, ky)  # [B, 2, H, W]

    pred_norm = torch.linalg.norm(pred_grad, dim=1, keepdim=True).clamp_min(eps)
    gt_norm = torch.linalg.norm(gt_grad, dim=1, keepdim=True).clamp_min(eps)

    # Cosine similarity
    denom = pred_norm * gt_norm + eps
    cos_sim = (pred_grad * gt_grad).sum(dim=1, keepdim=True) / denom
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Edge mask M from GT gradient percentile scaling
    gt_detached = gt_norm.detach()
    g_p95 = torch.quantile(gt_detached[..., ::4, ::4], 0.95).clamp(min=eps)
    grad_scale = g_p95 / 0.7 + eps
    mask = torch.clamp(gt_detached / grad_scale, 0.0, 1.0)

    mask_sum = mask.sum().clamp_min(1.0)
    flat_mask = 1.0 - mask
    flat_sum = flat_mask.sum().clamp_min(1.0)

    loss_cos = (mask * (1.0 - cos_sim)).sum() / mask_sum
    loss_flat = flat_weight * (flat_mask * pred_norm).sum() / flat_sum
    loss = loss_cos + loss_flat

    if return_components:
        edge_term = (mask * (1.0 - cos_sim)).mean().detach()
        flat_term = (flat_mask * pred_norm).mean().detach()
        return loss, edge_term, flat_term
    return loss


def gradient_loss(pred_rgb, gt_rgb=None, gt_grad_cached=None, gt_image_id=None, valid_mask=None, flat_weight=0.5, return_components=False):
    """
    Unified edge-aware gradient loss (GlowGS innovation #2).
    
    Design Philosophy:
    - Addresses 3DGS texture over-smoothing by explicitly supervising image gradients.
    - Uses GT Sobel gradients to build an adaptive edge confidence map.
    - Edge regions (high GT gradient): enforce gradient alignment to restore sharp edges.
    - Flat regions (low GT gradient): penalize spurious high-frequency noise.
    - Single unified formulation with adjustable flat_weight (alpha) for clean API.
    
    Key Difference from naive gradient matching:
    - Spatially-adaptive weighting based on GT edge strength prevents flat regions
      from being dominated by alignment loss, which would amplify noise.
    
    Args:
        pred_rgb: Predicted image [C, H, W] or [B, C, H, W]
        gt_rgb: Ground truth image [C, H, W] or [B, C, H, W] (optional if gt_grad_cached provided)
        gt_grad_cached: Precomputed GT Sobel gradients [2, H, W] for performance
        gt_image_id: Unique identifier for automatic caching (e.g., camera uid)
        valid_mask: Optional binary mask [H, W] or [B, 1, H, W] for valid regions
        flat_weight: Weight for flat region regularization term (alpha in paper)
        return_components: If True, also return (edge_term, flat_term) for logging
    
    Returns:
        loss: Scalar edge-aware gradient loss
        (optional) edge_term_mean, flat_term_mean: Component values for debugging
    """
    # Ensure 4D input: [B, C, H, W]
    if pred_rgb.dim() == 3:
        pred_rgb = pred_rgb.unsqueeze(0)
    
    # Get cached Sobel kernels
    kx, ky = _get_sobel_kernels(pred_rgb.device, pred_rgb.dtype)
    
    # Convert to grayscale for gradient computation
    pred_gray = _rgb_to_grayscale(pred_rgb)
    
    # Compute prediction Sobel magnitude
    G_pred = _compute_sobel_magnitude(pred_gray, kx, ky)
    
    # OPTIMIZATION: Use cached GT gradients if available
    if gt_grad_cached is not None:
        # Compute magnitude from cached gradients [2, H, W] -> [1, 1, H, W]
        gt_grad_expanded = gt_grad_cached.unsqueeze(0)  # [1, 2, H, W]
        G_gt = torch.sqrt(gt_grad_expanded[:, 0:1, :, :] ** 2 + gt_grad_expanded[:, 1:2, :, :] ** 2 + 1e-6)
    elif gt_image_id is not None and gt_rgb is not None:
        # Automatic caching using image_id
        if gt_rgb.dim() == 3:
            gt_rgb = gt_rgb.unsqueeze(0)
        gt_grad = _get_or_compute_gt_gradients(gt_rgb, image_id=gt_image_id)
        G_gt = torch.sqrt(gt_grad[:, 0:1, :, :] ** 2 + gt_grad[:, 1:2, :, :] ** 2 + 1e-6)
    else:
        # Fallback: compute on-the-fly
        if gt_rgb is None:
            raise ValueError("Either gt_rgb or gt_grad_cached must be provided")
        if gt_rgb.dim() == 3:
            gt_rgb = gt_rgb.unsqueeze(0)
        gt_gray = _rgb_to_grayscale(gt_rgb)
        G_gt = _compute_sobel_magnitude(gt_gray, kx, ky)
    
    # Normalize gradients to [0, 1] range for stable loss magnitude
    # Sobel max response on [0,1] image is ~4.0; use 4.0 as normalizer
    grad_normalizer = 4.0
    G_pred_norm = G_pred / grad_normalizer
    G_gt_norm = G_gt / grad_normalizer
    
    # Build edge confidence mask from GT gradient (detached to avoid gradient path issues)
    # Higher values at edges, lower at flat regions
    # Use adaptive percentile-based scaling for robustness across different scene types
    eps = 1e-6
    G_gt_detached = G_gt_norm.detach()
    # Use 95th percentile as reference scale (more robust than mean for bimodal distributions)
    # Subsample by stride=4 to reduce quantile sorting cost (~16x speedup, negligible accuracy loss)
    g_p95 = torch.quantile(G_gt_detached[..., ::4, ::4], 0.95).clamp(min=eps)
    # Scale so that p95 maps to ~0.7, allowing headroom for stronger edges
    grad_scale = g_p95 / 0.7 + eps
    edge_confidence = torch.clamp(G_gt_detached / grad_scale, 0.0, 1.0)
    
    # Edge alignment term: match pred gradient to GT gradient at edge locations
    edge_term = torch.abs(G_pred_norm - G_gt_norm) * edge_confidence
    
    # SAFE VERSION: Removed flat regularization term (flat_term)
    # Original implementation suppressed gradients in non-edge regions, causing texture over-smoothing
    # Now we only enforce edge alignment without suppressing mid-frequency details
    
    # Apply valid mask if provided
    if valid_mask is not None:
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        edge_term = edge_term * valid_mask
        # Compute mean only over valid pixels
        num_valid = valid_mask.sum().clamp(min=1.0)
        edge_term_mean = edge_term.sum() / num_valid
        # Dummy values for backward compatibility with return_components
        flat_term_mean = torch.tensor(0.0, device=edge_term.device)
    else:
        edge_term_mean = edge_term.mean()
        # Dummy values for backward compatibility with return_components
        flat_term_mean = torch.tensor(0.0, device=edge_term.device)
    
    # Safe loss: only edge alignment, no suppression
    loss = edge_term_mean
    
    if return_components:
        return loss, edge_term_mean.detach(), flat_term_mean.detach()
    return loss

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_raw(img1, img2, window_size=11):
    """
    Compute per-pixel SSIM dissimilarity map (1 - ssim_map).
    Returns a [B, 1, H, W] tensor for element-wise weighting.
    Used by background-weighted loss to emphasize peripheral regions.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return dissimilarity map (1 - ssim), averaged over channels → [B, 1, H, W]
    dssim_map = 1.0 - ssim_map.mean(dim=1, keepdim=True)
    return dssim_map

