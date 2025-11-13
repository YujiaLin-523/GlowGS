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

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import HybridEncoderParams, WaveletLossParams, TrainingAccelParams
from arguments import VMauxLossParams, OrthogonalityParams, NumericStabilityParams, LocalityLossParams, FeatureRegParams
from arguments import KappaAdaptiveParams, AnisotropyConstraintParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# [HYBRID] Enable TF32 for faster matmul on Ampere+ GPUs (FP32 with accelerated GEMM)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')  # PyTorch 2.0+ for TF32
except AttributeError:
    pass  # Older PyTorch versions don't have this


# ============================================================================
# VM AUXILIARY LOSS (Low-frequency proxy loss, VM-only backprop)
# ============================================================================
def lowpass_image(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply spatial low-pass filter to image using average pooling.

    Args:
        img: [B, H, W, C] or [H, W, C] image tensor
        kernel_size: Odd kernel size for average pooling

    Returns:
        Low-pass filtered image (same shape as input)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # [1, H, W, C]

    # Convert to [B, C, H, W] for conv2d
    img_t = img.permute(0, 3, 1, 2)
    C = img_t.shape[1]

    # Create average pooling kernel [C, 1, K, K] for depthwise conv
    pad = kernel_size // 2
    weight = torch.ones(C, 1, kernel_size, kernel_size, device=img.device, dtype=img.dtype) / (kernel_size * kernel_size)

    img_lp = torch.nn.functional.conv2d(img_t, weight, padding=pad, groups=C)

    # Convert back to [B, H, W, C]
    return img_lp.permute(0, 2, 3, 1).squeeze(0) if img.shape[0] == 1 else img_lp.permute(0, 2, 3, 1)


def compute_vm_aux_loss(gaussians, gt_image, vm_aux_config, iteration):
    """
    Compute VM auxiliary loss: low-frequency proxy loss that only backprops to VM branch.

    Creates a lightweight projection head (vm_feat -> RGB) and compares against
    low-pass filtered GT image. This provides stable gradients to VM planes even
    when hash branch dominates main loss.

    Args:
        gaussians: GaussianModel with encoder containing _vm_feat_for_aux buffer
        gt_image: [3, H, W] or [H, W, 3] ground truth image
        vm_aux_config: VMauxLossParams object
        iteration: Current training iteration (for kernel size curriculum)

    Returns:
        Scalar VM auxiliary loss (0 if disabled)
    """
    if not vm_aux_config or not vm_aux_config.vm_aux_enable:
        return gt_image.new_zeros([])

    if not hasattr(gaussians, '_grid') or not hasattr(gaussians._grid, '_vm_feat_for_aux'):
        return gt_image.new_zeros([])

    # [FIX A-1] Read from _vm_feat_for_aux (no detach, gradients flow to VM)
    vm_feat = gaussians._grid._vm_feat_for_aux  # [N, D]
    if vm_feat is None or vm_feat.numel() == 0:
        return gt_image.new_zeros([])

    # Curriculum: anneal blur kernel size over training
    t = min(1.0, iteration / max(1, vm_aux_config.vm_aux_blur_steps))
    k = int(round(vm_aux_config.vm_aux_blur_k_init + (vm_aux_config.vm_aux_blur_k_end - vm_aux_config.vm_aux_blur_k_init) * t))
    k = max(3, k | 1)  # Ensure odd kernel size >= 3

    # Low-pass filter GT image
    if gt_image.ndim == 3 and gt_image.shape[0] == 3:
        gt_image = gt_image.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]

    with torch.no_grad():
        gt_lp = lowpass_image(gt_image, k)  # [H, W, 3]

    # Project VM features to RGB (simplified: aggregate to mean)
    # NOTE: For proper implementation, should aggregate per-pixel via splatting/rasterization
    # This is a minimal-change placeholder that still provides gradient signal to VM
    vm_rgb = gaussians._grid.head_vm(vm_feat)  # [N, 3]
    vm_rgb_mean = vm_rgb.mean(dim=0)  # [3] scalar mean per channel

    # Broadcasting will handle shape mismatch automatically in loss computation
    gt_lp_mean = gt_lp.mean(dim=(0, 1))  # [3] mean per channel

    # L1 loss between VM mean RGB and GT low-pass mean (efficient, no expand)
    loss_vm = torch.nn.functional.l1_loss(vm_rgb_mean, gt_lp_mean) * vm_aux_config.vm_aux_lambda

    # [FIX A-1] Clear ephemeral VM feature cache after use to prevent graph retention across steps
    gaussians._grid._vm_feat_for_aux = None

    return loss_vm


# ============================================================================
# FEATURE REGULARIZATION LOSS (Encoder feature norm penalty)
# ============================================================================
def compute_feature_reg_loss(gaussians, feat_reg_config, iteration):
    """
    Compute feature norm regularization to prevent encoder overfitting.
    
    Penalizes large L2 norms of encoder output features, encouraging compact
    representations that generalize better to test views.
    
    Args:
        gaussians: GaussianModel with _grid encoder
        feat_reg_config: FeatureRegParams object
        iteration: Current training iteration
    
    Returns:
        Scalar feature regularization loss (0 if disabled or before start iter)
    """
    if not feat_reg_config or not feat_reg_config.feat_reg_enable:
        return torch.tensor(0.0, device='cuda')
    
    # Wait until training stabilizes before applying regularization
    if iteration < feat_reg_config.feat_reg_start:
        return torch.tensor(0.0, device='cuda')
    
    # Sample a subset of points to compute feature norm (avoid full forward pass overhead)
    N = gaussians._xyz.shape[0]
    if N == 0:
        return torch.tensor(0.0, device='cuda')
    
    # Sample 10% of points or max 5000 points
    sample_size = min(max(N // 10, 1000), 5000)
    idx = torch.randperm(N, device='cuda')[:sample_size]
    
    # Get encoder features for sampled points
    xyz_sample = gaussians.get_contracted_xyz[idx]
    
    # Forward through encoder only (no heads)
    if hasattr(gaussians, '_grid'):
        feats = gaussians._grid(xyz_sample)  # [sample_size, feat_dim]
        
        # L2 norm penalty: encourage small feature magnitudes
        feat_norm = torch.norm(feats, p=2, dim=-1).mean()
        
        return feat_norm * feat_reg_config.feat_reg_lambda
    
    return torch.tensor(0.0, device='cuda')


# ============================================================================
# LOCALITY LOSS (Neighboring Gaussians' attribute similarity)
# ============================================================================
def compute_locality_loss(gaussians, visibility_filter, loc_config):
    """
    Compute locality regularization loss on visible Gaussians with stop-grad neighbors.
    
    Encodes the LocoGS prior: spatially close Gaussians should have similar attributes
    to improve generalization on sparse-view data. Uses top-k neighbors in [0,1]^3
    contracted space with Gaussian-weighted attribute difference penalties.
    
    [FIX C] Key improvements:
    - Stop-grad on neighbor attributes: only anchor Gaussians receive gradients
    - Quaternion geometric distance: 1 - |dot(qi,qj)| for proper rotation metric
    - All topology (KNN indices, weights) computed under no_grad for efficiency
    
    Args:
        gaussians: GaussianModel with attribute tensors
        visibility_filter: [N] boolean mask of visible points in current view
        loc_config: LocalityLossParams object
    
    Returns:
        Scalar locality loss (0 if disabled or M<2)
    """
    if not loc_config or not loc_config.loc_enable:
        return torch.tensor(0.0, device='cuda')
    
    # Extract visible indices
    idx = visibility_filter.nonzero(as_tuple=True)[0]
    M = idx.numel()
    
    if M < 2:
        # Need at least 2 points to compute neighbors
        return torch.tensor(0.0, device='cuda')
    
    # Memory safety: cap M to max_points via random subsampling
    if M > loc_config.max_points:
        perm = torch.randperm(M, device=idx.device)[:loc_config.max_points]
        idx = idx[perm]
        M = loc_config.max_points
    
    # Use contracted positions in [0,1]^3 (consistent with encoder coordinate unification)
    xyz = gaussians.get_contracted_xyz[idx]  # [M, 3]
    
    # [FIX C] Compute KNN topology under no_grad (pure geometry, no backprop)
    with torch.no_grad():
        # Compute pairwise distances: [M, M]
        dists = torch.cdist(xyz, xyz, p=2)  # Euclidean distance
        
        # Top-k neighbors (exclude self by setting diagonal to inf before topk)
        dists_masked = dists.clone()
        dists_masked.fill_diagonal_(float('inf'))
        
        k = min(loc_config.loc_k, M - 1)  # Exclude self, cap at M-1
        topk_dists, topk_indices = torch.topk(dists_masked, k=k, dim=-1, largest=False)  # [M, k]
        
        # Gaussian spatial weighting: W = exp(-d^2 / (2*sigma^2))
        sigma2 = loc_config.loc_sigma ** 2
        weights = torch.exp(-topk_dists ** 2 / (2.0 * sigma2))  # [M, k]
    
    # Gather anchor attributes (these WILL receive gradients)
    opacity = gaussians.get_opacity[idx]  # [M, 1]
    scale = gaussians.scaling_activation(gaussians._scaling[idx])  # [M, 3], normalized scales
    rotation = gaussians.get_rotation[idx]  # [M, 4], unit quaternions
    
    # Compute attribute differences with neighbors
    loss_total = 0.0
    
    # 1. Opacity similarity (stop-grad on neighbors)
    if loc_config.w_opacity > 0:
        opacity_nbr = gaussians.get_opacity[idx[topk_indices]].detach()  # [M, k, 1], no grad
        diff_opacity = (opacity.unsqueeze(1) - opacity_nbr).abs()  # [M, k, 1]
        loss_opacity = (weights.unsqueeze(-1) * diff_opacity).mean()
        loss_total += loc_config.w_opacity * loss_opacity
    
    # 2. Scale similarity (stop-grad on neighbors)
    if loc_config.w_scale > 0:
        scale_nbr = gaussians.scaling_activation(gaussians._scaling[idx[topk_indices]]).detach()  # [M, k, 3], no grad
        diff_scale = (scale.unsqueeze(1) - scale_nbr).pow(2).sum(dim=-1)  # [M, k], L2
        loss_scale = (weights * diff_scale).mean()
        loss_total += loc_config.w_scale * loss_scale
    
    # 3. Rotation similarity using quaternion geometric distance (stop-grad on neighbors)
    # [FIX C] Use 1 - |dot(qi, qj)| instead of L2 distance for proper rotation metric
    if loc_config.w_rot > 0:
        rotation_nbr = gaussians.get_rotation[idx[topk_indices]].detach()  # [M, k, 4], no grad
        rot_expand = rotation.unsqueeze(1)  # [M, 1, 4]
        
        # Compute dot product: high |dot| => similar orientation (antipodal symmetry handled)
        dot = (rot_expand * rotation_nbr).sum(dim=-1)  # [M, k]
        
        # Quaternion geometric distance: d = 1 - |dot(qi, qj)|
        # This naturally handles antipodal symmetry (q ≈ -q) and is bounded in [0, 1]
        dist_quat = 1.0 - dot.abs()  # [M, k], range [0, 1]
        loss_rot = (weights * dist_quat).mean()
        loss_total += loc_config.w_rot * loss_rot
    
    # 4. SH residual similarity (stop-grad on neighbors)
    if loc_config.w_sh > 0 and gaussians.max_sh_degree > 0:
        sh_rest = gaussians._features_rest[idx]  # [M, (deg+1)^2-1, 3]
        sh_rest_flat = sh_rest.view(M, -1)  # [M, D]
        sh_nbr = gaussians._features_rest[idx[topk_indices]].detach().view(M, k, -1)  # [M, k, D], no grad
        diff_sh = (sh_rest_flat.unsqueeze(1) - sh_nbr).pow(2).sum(dim=-1)  # [M, k], L2
        loss_sh = (weights * diff_sh).mean()
        loss_total += loc_config.w_sh * loss_sh
    
    return loss_total * loc_config.loc_lambda


# ============================================================================
# ORTHOGONALITY LOSS (Feature decorrelation between VM and Hash branches)
# ============================================================================
def compute_orth_loss(gaussians, orth_config):
    """
    Compute feature decorrelation loss between VM and Hash branches.

    Encourages VM and Hash to capture complementary information by penalizing
    high cosine similarity between their feature outputs.

    Args:
        gaussians: GaussianModel with encoder containing _last_vm_feat and _last_hash_feat buffers
        orth_config: OrthogonalityParams object

    Returns:
        Scalar orthogonality loss (0 if disabled)
    """
    if not orth_config or not orth_config.orth_enable:
        return torch.tensor(0.0, device='cuda')

    if not hasattr(gaussians, '_grid'):
        return torch.tensor(0.0, device='cuda')

    vm_feat = getattr(gaussians._grid, '_last_vm_feat', None)
    hash_feat = getattr(gaussians._grid, '_last_hash_feat', None)

    if vm_feat is None or hash_feat is None or vm_feat.numel() == 0 or hash_feat.numel() == 0:
        return torch.tensor(0.0, device='cuda')

    # Center features (remove mean)
    vm_centered = vm_feat - vm_feat.mean(dim=0, keepdim=True)
    hash_centered = hash_feat - hash_feat.mean(dim=0, keepdim=True)

    # Compute mean cosine similarity
    numerator = (vm_centered * hash_centered).sum(dim=-1).mean()
    denominator = (vm_centered.norm(dim=-1).mean() * hash_centered.norm(dim=-1).mean() + 1e-8)

    cos_sim = (numerator / denominator).abs()

    return cos_sim * orth_config.orth_lambda


# ============================================================================
# WAVELET LOSS (Optional, image-domain high-frequency emphasis)
# ============================================================================
def compute_wavelet_loss(pred_img, gt_img, wavelet_config, iteration=0):
    """
    Optional wavelet decomposition loss for high-frequency details.

    Decomposes images into subbands (LL, LH, HL, HH) via DWT and applies
    weighted L1 loss. Emphasizes high-frequency reconstruction without
    modifying renderer or encoder internals.

    Args:
        pred_img: [H, W, 3] predicted image tensor
        gt_img: [H, W, 3] ground truth image tensor
        wavelet_config: WaveletLossParams object
        iteration: Current training iteration (for ramping schedule)

    Returns:
        Scalar wavelet loss (0 if disabled or unavailable)
    """
    if not wavelet_config or not wavelet_config.wavelet_enable:
        return pred_img.new_zeros([])

    try:
        # Try importing pytorch_wavelets (optional dependency)
        from pytorch_wavelets import DWTForward
    except ImportError:
        # Graceful fallback: disable silently if not installed
        return pred_img.new_zeros([])

    # [CHECKLIST A8] Gradual ramp: avoid conflict with VM LPF + gate transition
    # Linear ramp from start_iter to ramp_iter
    if iteration < wavelet_config.wavelet_start_iter:
        return pred_img.new_zeros([])
    elif iteration < wavelet_config.wavelet_ramp_iter:
        # Linear ramp: 0 at start, full lambda_h at ramp_iter
        progress = (iteration - wavelet_config.wavelet_start_iter) / (wavelet_config.wavelet_ramp_iter - wavelet_config.wavelet_start_iter)
        lambda_h_weight = progress
    else:
        # Full strength after ramp_iter
        lambda_h_weight = 1.0

    # Convert to grayscale for stable wavelet decomposition
    if wavelet_config.wavelet_grayscale and pred_img.shape[-1] == 3:
        # RGB to Y (luminance) using ITU-R BT.601 coefficients
        pred_y = (0.299 * pred_img[..., 0] +
                  0.587 * pred_img[..., 1] +
                  0.114 * pred_img[..., 2]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        gt_y = (0.299 * gt_img[..., 0] +
                0.587 * gt_img[..., 1] +
                0.114 * gt_img[..., 2]).unsqueeze(0).unsqueeze(0)
    else:
        pred_y = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        gt_y = gt_img.permute(2, 0, 1).unsqueeze(0)

    # Wavelet decomposition
    dwt = DWTForward(J=wavelet_config.wavelet_levels, wave='haar', mode='symmetric').to(pred_img.device)
    Yc_pred, Yh_pred = dwt(pred_y)  # LL, [list of (LH, HL, HH)]
    Yc_gt, Yh_gt = dwt(gt_y)

    # L1 loss on low-frequency (LL)
    ll_loss = torch.mean(torch.abs(Yc_pred - Yc_gt))

    # L1 loss on high-frequency subbands (LH, HL, HH)
    hf_loss = 0.0
    for level in range(len(Yh_pred)):
        for band in range(3):  # LH, HL, HH
            hf_loss += torch.mean(torch.abs(Yh_pred[level][:, band] - Yh_gt[level][:, band]))

    # Weighted combination with ramped high-frequency weight
    total_loss = wavelet_config.wavelet_lambda_ll * ll_loss + (lambda_h_weight * wavelet_config.wavelet_lambda_h) * hf_loss
    return total_loss


# ============================================================================
# HELPER: Gradient Norm Logging
# ============================================================================
def get_grad_norm(param):
    """Safely extract gradient norm from a parameter."""
    if param is not None and param.grad is not None:
        return float(param.grad.norm().detach())
    return 0.0


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             hybrid_config=None, vm_aux_config=None, orth_config=None, numeric_config=None, wavelet_config=None, accel_config=None, loc_config=None, feat_reg_config=None, kappa_config=None, aniso_config=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, hybrid_config=hybrid_config, aniso_config=aniso_config)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # [CHECKLIST 1] Two-stage adaptive kappa initialization (universal benefit for all scenes)
    # Stage 1: Cold-start (iter 0-500): weak VM+Hash, prevent early gradient instability
    # Stage 2: Statistics collection (iter ~1000): set kappa based on log2(sigma_max) median
    kappa_adaptive_initialized = False
    sigma_max_history = []  # Collect sigma_max for statistics
    
    # During warmup, use stronger VM floor to prevent early gradient starvation
    if kappa_config and kappa_config.kappa_adaptive_enable and hybrid_config and hybrid_config.hybrid_enable:
        if first_iter < kappa_config.kappa_warmup_iter:
            original_beta_min = hybrid_config.beta_min_start
            hybrid_config.beta_min_start = kappa_config.kappa_beta_min_warmup
            print(f"[KAPPA-ADAPTIVE] Warmup phase (0-{kappa_config.kappa_warmup_iter}): β_min={kappa_config.kappa_beta_min_warmup:.2f}")
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # --- Optimizer debug（不影响训练性能）---
    if hasattr(gaussians, "optimizer_i"):
        total_groups = len(gaussians.optimizer_i.param_groups)
        total_params = sum(len(g["params"]) for g in gaussians.optimizer_i.param_groups)
        print(f"[DEBUG-OPTIMIZER] Total param_groups in optimizer_i: {total_groups}, total params: {total_params}")
        # 尝试识别 VM 分组
        vm_cnt = 0
        for g in gaussians.optimizer_i.param_groups:
            name = g.get("name", "")
            if "vm" in name or "VM" in name:
                vm_cnt += len(g["params"])
        if vm_cnt > 0:
            print(f"[DEBUG-OPTIMIZER] VM encoder group found: {vm_cnt} parameters")
            print(f"[DEBUG-OPTIMIZER] ✓ VM parameters correctly registered in optimizer")
        else:
            print(f"[DEBUG-OPTIMIZER] (Note) Could not find named VM group; ensure VM params are in optimizer_i")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # [FIX] AMP DISABLED for stability (use FP32 + TF32 instead)
    assert not accel_config.amp_enable, "AMP must be disabled for hybrid encoder stability (use TF32 instead)"

    # [HYBRID] Optional torch.compile for encoder ONLY (experimental)
    if accel_config.torch_compile and hasattr(gaussians, '_grid'):
        try:
            gaussians._grid = torch.compile(
                gaussians._grid,
                mode="reduce-overhead",
                fullgraph=False
            )
            print("[HYBRID] torch.compile enabled for encoder")
        except Exception as e:
            print(f"[HYBRID] torch.compile failed: {e}, continuing without it")

    # [FIX] Print configuration and memory stats at training start
    enc_type = "hybrid(vm+hash)" if (hybrid_config and hybrid_config.hybrid_enable) else "hash-only"

    # Memory diagnostic
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"\n[GPU MEMORY]")
        print(f"  Allocated: {gpu_mem_allocated:.2f} GB")
        print(f"  Reserved:  {gpu_mem_reserved:.2f} GB")
        print(f"  Total:     {gpu_mem_total:.2f} GB")
        print(f"  Available: {gpu_mem_total - gpu_mem_reserved:.2f} GB")

    print(f"\n[TRAINING CONFIG]")
    print(f"  Encoder: {enc_type}")
    if hybrid_config and hybrid_config.hybrid_enable:
        print(f"  VM: rank={hybrid_config.vm_rank}, res={hybrid_config.vm_plane_res}, out_dim={hybrid_config.vm_out_dim}")
        print(f"  Fusion: convex blend, hash_gain {hybrid_config.init_hash_gain:.2f}→1.0 over {hybrid_config.warm_steps} steps")
        print(f"  Gate: α {hybrid_config.gate_alpha_start:.1f}→{hybrid_config.gate_alpha_end:.1f}, τ {hybrid_config.gate_tau_start:.1f}→{hybrid_config.gate_tau_end:.1f}")
        print(f"  VM LR multiplier: {hybrid_config.vm_lr_multiplier:.1f}x (no weight decay)")
        # [NEW E] Print micro-batching strategy (batching only when memory-constrained)
        micro_batch_size = hybrid_config.update_batch_size if hasattr(hybrid_config, 'update_batch_size') else 0
        oom_threshold = hybrid_config.oom_threshold if hasattr(hybrid_config, 'oom_threshold') else float('inf')
        if micro_batch_size > 0 and oom_threshold < float('inf'):
            print(f"  Micro-batch: auto (size={micro_batch_size}, trigger at N>{oom_threshold/1e6:.1f}M points)")
        else:
            print(f"  Micro-batch: disabled (full-batch training)")
    # 更清晰的 AMP 打印
    print(f"  AMP: False (FP32+TF32)")
    print(f"  Grad clip: {numeric_config.grad_clip_norm if numeric_config.grad_clip_norm > 0 else 'disabled'}")
    # [NEW E] Print locality loss config with annealing
    loc_enabled = loc_config and loc_config.loc_enable
    if loc_enabled:
        print(f"  Locality: λ={loc_config.loc_lambda}→{loc_config.loc_lambda_end} (anneal: {loc_config.loc_anneal_start}→{loc_config.loc_anneal_end}), k={loc_config.loc_k}")
    else:
        print(f"  Locality loss: disabled")
    # [GENERALIZATION] Print feature reg config
    feat_reg_enabled = feat_reg_config and feat_reg_config.feat_reg_enable
    if feat_reg_enabled:
        print(f"  Feature reg: λ={feat_reg_config.feat_reg_lambda}, start={feat_reg_config.feat_reg_start}")
    else:
        print(f"  Feature reg: disabled")
    # [WAVELET] Print wavelet loss config
    wavelet_enabled = wavelet_config and wavelet_config.wavelet_enable
    if wavelet_enabled:
        print(f"  Wavelet: λ_HF={wavelet_config.wavelet_lambda_h}, λ_LL={wavelet_config.wavelet_lambda_ll}, start={wavelet_config.wavelet_start_iter}")
    else:
        print(f"  Wavelet loss: disabled")
    # [3DGS BASELINE] Print densification/pruning parameters with fixes
    print(f"  Densify: grad_thr={opt.densify_grad_threshold:.1e}, until_iter={opt.densify_until_iter}, interval={opt.densification_interval}, size_thr=16 (was 20)")
    print(f"  Prune: opacity_thr={opt.prune_opacity_threshold:.3f}, interval={opt.prune_interval} (after densify stops)")
    print(f"  SH upgrade: every 500 iters (accelerated, was 1000)")
    print(f"  Log every: {accel_config.log_every} steps")
    print(f"{'='*80}\n")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # [HYBRID FIX] GUI rendering should not build gradient graph
                    # to avoid "backward through graph second time" error with tinycudann
                    with torch.no_grad():
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # [CHECKLIST 1] Two-stage kappa: collect sigma_max stats until 1k iter for reliable gating
        if kappa_config and kappa_config.kappa_adaptive_enable and not kappa_adaptive_initialized:
            if iteration <= kappa_config.kappa_init_iter:
                # Stage 1: Collect statistics during first 1k iterations
                with torch.no_grad():
                    scaling = gaussians.get_scaling  # [N, 3]
                    if scaling.numel() > 0:
                        sigma_max = scaling.max(dim=-1)[0]  # [N]
                        sigma_max_history.append(sigma_max.detach().cpu())
                
                # Stage 2: At iteration 1k, set kappa once based on statistics
                if iteration == kappa_config.kappa_init_iter and len(sigma_max_history) > 0:
                    sigma_max_all = torch.cat(sigma_max_history, dim=0)  # Stack all samples
                    
                    # 使用 numpy 或排序法计算分位数，避免 torch.quantile 的内存问题
                    # 方案：对大张量使用 sort + indexing，内存友好
                    sigma_sorted, _ = torch.sort(sigma_max_all)
                    n = sigma_sorted.shape[0]
                    
                    median_sigma = float(sigma_sorted[n // 2])
                    p10_sigma = float(sigma_sorted[int(n * 0.1)])
                    p90_sigma = float(sigma_sorted[int(n * 0.9)])
                    
                    # Set kappa based on median + delta offset
                    delta = kappa_config.kappa_delta
                    adaptive_kappa = 2.0 ** (torch.log2(torch.tensor(median_sigma)) + delta)
                    adaptive_kappa = float(adaptive_kappa.item())
                    
                    # Update hybrid config with computed kappa
                    hybrid_config.gate_kappa = adaptive_kappa
                    
                    # Restore original beta_min after warmup
                    hybrid_config.beta_min_start = original_beta_min
                    
                    print(f"\n[KAPPA-ADAPTIVE {iteration}] Stage 2 - Initialized:")
                    print(f"  Samples: {len(sigma_max_all)} Gaussians")
                    print(f"  σ_max stats: median={median_sigma:.4f}, p10={p10_sigma:.4f}, p90={p90_sigma:.4f}")
                    print(f"  κ = {adaptive_kappa:.4f} (Δ={delta:+.1f})")
                    print(f"  Expected log₂(κ/σ) @ median: {torch.log2(torch.tensor(adaptive_kappa/median_sigma)):.2f}\n")
                    
                    kappa_adaptive_initialized = True
                    sigma_max_history = []  # Clear history
        # Original: every 1000 iters → 10k to reach deg=3
        # New: every 500 iters → 5k to reach deg=3 (especially important for specular surfaces)
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # [FIX] Annealing schedules (hash_gain, gate, beta_min) BEFORE forward pass
        if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'set_hash_gain'):
            t = min(1.0, iteration / max(1, hybrid_config.warm_steps))  # [0, 1]

            # hash_gain: 0.05 → 1.0 over warm_steps
            hash_gain = hybrid_config.init_hash_gain + (1.0 - hybrid_config.init_hash_gain) * t
            gaussians._grid.set_hash_gain(hash_gain)

            # gate schedule: alpha (3→8), tau (1→0) over warm_steps
            gate_alpha = hybrid_config.gate_alpha_start + (hybrid_config.gate_alpha_end - hybrid_config.gate_alpha_start) * t
            gate_tau = hybrid_config.gate_tau_start + (hybrid_config.gate_tau_end - hybrid_config.gate_tau_start) * t
            gaussians._grid.set_gate_sched(gate_alpha, gate_tau)

            # beta_min: 0.10 → 0.05 over warm_steps
            beta_min = hybrid_config.beta_min_start + (hybrid_config.beta_min_end - hybrid_config.beta_min_start) * t
            gaussians._grid.set_beta_min(beta_min)
        
        # [FIX B] VM LPF weakening schedule: restore high frequencies gradually
        # Weakens kernel size (3→1) then fully disables LPF for sharp edge recovery
        if (hybrid_config and hybrid_config.hybrid_enable and 
            hasattr(gaussians, '_grid') and hasattr(gaussians._grid, 'vm')):
            vm = gaussians._grid.vm
            if hybrid_config.vm_lpf_weaken_steps > 0:
                t_lpf = iteration / float(hybrid_config.vm_lpf_weaken_steps)
                if t_lpf < 1.0:
                    # Gradually reduce kernel size: 3 → 2 → 1
                    new_k = int(round(3.0 - 2.0 * t_lpf))
                    vm.lpf_kernel = max(1, min(3, new_k))  # Clamp to [1, 3]
                else:
                    # Fully disable LPF after weaken_steps
                    vm.lpf_enable = False
                    
                # [ACCEPTANCE CHECK B] Log LPF schedule (once per 1k iterations)
                if iteration % 1000 == 0:
                    lpf_status = f"kernel={vm.lpf_kernel}" if vm.lpf_enable else "DISABLED"
                    print(f"[LPF-SCHEDULE {iteration:05d}] {lpf_status}")

        # [FIX] Forward pass in FP32 (NO AMP, use TF32 for performance)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss computation: main pixel loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        pixel_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # [NEW] VM auxiliary loss (low-frequency proxy, VM-only backprop)
        loss_vm_aux = compute_vm_aux_loss(gaussians, gt_image, vm_aux_config, iteration)

        # [NEW] Orthogonality loss (feature decorrelation between VM and Hash)
        loss_orth = compute_orth_loss(gaussians, orth_config)

        # [NEW D] Locality loss with annealing schedule (neighboring Gaussians' attribute similarity)
        # [FIX CHECKLIST A7] Anneal loc_lambda to avoid late-stage over-smoothing
        if loc_config and loc_config.loc_enable:
            if iteration < loc_config.loc_anneal_start:
                current_loc_lambda = loc_config.loc_lambda
            elif iteration < loc_config.loc_anneal_end:
                t = (iteration - loc_config.loc_anneal_start) / (loc_config.loc_anneal_end - loc_config.loc_anneal_start)
                current_loc_lambda = loc_config.loc_lambda + (loc_config.loc_lambda_end - loc_config.loc_lambda) * t
            else:
                current_loc_lambda = loc_config.loc_lambda_end
            
            # Temporarily override loc_lambda for this iteration
            original_lambda = loc_config.loc_lambda
            loc_config.loc_lambda = current_loc_lambda
            loss_loc = compute_locality_loss(gaussians, visibility_filter, loc_config)
            loc_config.loc_lambda = original_lambda  # Restore
        else:
            loss_loc = torch.tensor(0.0, device='cuda')
        
        # [ACCEPTANCE CHECK C] Log locality loss components (once per 1k iterations)
        if loc_config and loc_config.loc_enable and iteration % 1000 == 0:
            visible_count = visibility_filter.sum().item()
            print(f"[LOCALITY {iteration:05d}] loss={float(loss_loc):.5f} visible={visible_count} "
                  f"λ={current_loc_lambda:.4f}(annealing) k={loc_config.loc_k}")

        # [GENERALIZATION] Feature regularization loss (encoder norm penalty)
        loss_feat_reg = compute_feature_reg_loss(gaussians, feat_reg_config, iteration)

        # [NEW CHECKLIST 2] Anisotropy constraint loss (hard clamp + soft regularizer)
        loss_aniso = gaussians.compute_aniso_regularizer(iteration)

        # [HYBRID] Optional wavelet loss (image domain high-frequency)
        # [FIX CHECKLIST A8] Delay wavelet loss start to avoid VM LPF conflict
        if wavelet_config and wavelet_config.wavelet_enable and iteration >= wavelet_config.wavelet_start_iter:
            wavelet_loss = compute_wavelet_loss(image.permute(1, 2, 0), gt_image.permute(1, 2, 0), wavelet_config, iteration)
        else:
            wavelet_loss = torch.tensor(0.0, device='cuda')

        # Mask losses (original)
        mask_loss = torch.mean(gaussians.get_mask)
        sh_mask_loss = 0.0
        if iteration > opt.densify_until_iter:
            for degree in range(1, gaussians.active_sh_degree + 1):
                lambda_degree = (2 * degree + 1) / ((gaussians.max_sh_degree + 1) ** 2 - 1)
                sh_mask_loss += lambda_degree * torch.mean(gaussians.get_sh_mask[..., degree - 1])

        # Total loss
        loss = pixel_loss + opt.lambda_mask * mask_loss + opt.lambda_sh_mask * sh_mask_loss
        if wavelet_config and wavelet_config.wavelet_enable and wavelet_loss.numel() > 0:
            loss = loss + wavelet_loss
        loss = loss + loss_vm_aux + loss_orth + loss_loc + loss_feat_reg + loss_aniso

        # [FIX] Backward in FP32 (NO AMP)
        loss.backward()

        # ========================================
        # [DEBUG] Gradient & Fusion Diagnostics (AFTER backward, BEFORE step)
        # ========================================
        try:
            from arguments import DEBUG_CFG
            if DEBUG_CFG.ON and iteration % DEBUG_CFG.EVERY == 0:
                import time
                step_start = time.time()

                # Helper: safe gradient norm
                def gnorm(param):
                    if param is None or param.grad is None:
                        return 0.0
                    return float(param.grad.norm())

                # 1) Gradient norms for key parameters (use object reference, not name matching)
                if DEBUG_CFG.GRAD_NORM and hybrid_config and hybrid_config.hybrid_enable:
                    vm_xy = gaussians._grid.vm.xy if hasattr(gaussians._grid, 'vm') else None
                    vm_mlp_weight = gaussians._grid.vm.mlp.weight if hasattr(gaussians._grid, 'vm') and hasattr(gaussians._grid.vm, 'mlp') else None
                    proj_hash_weight = gaussians._grid.proj_hash.weight if hasattr(gaussians._grid, 'proj_hash') else None

                    print(f"[DEBUG-GRAD {iteration:05d}] "
                          f"vm_xy={gnorm(vm_xy):.3e} "
                          f"vm_mlp={gnorm(vm_mlp_weight):.3e} "
                          f"proj_hash={gnorm(proj_hash_weight):.3e}")

                # 2) Fusion/Gate statistics (beta, gate, feature norms)
                # [CHECKLIST A1] Add sigma_max and log2(kappa/sigma) distribution monitoring
                if DEBUG_CFG.FUSION and hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, '_dbg_beta'):
                    beta = gaussians._grid._dbg_beta
                    gate = gaussians._grid._dbg_gate
                    vm_feat = gaussians._grid._dbg_vm
                    hash_feat = gaussians._grid._dbg_hash

                    def stats(x):
                        return (f"mean={float(x.mean()):.3f} std={float(x.std()):.3f} "
                                f"min={float(x.min()):.3f} max={float(x.max()):.3f}")
                    
                    def percentiles(x, name):
                        p10 = float(torch.quantile(x, 0.1))
                        p50 = float(torch.quantile(x, 0.5))
                        p90 = float(torch.quantile(x, 0.9))
                        return f"{name}: p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}"

                    print(f"[DEBUG-FUSION {iteration:05d}]")
                    print(f"  beta: {stats(beta.squeeze(-1))}")
                    print(f"  gate: {stats(gate)}")
                    print(f"  ||vm||={float(vm_feat.norm(dim=-1).mean()):.3f} "
                          f"||hash||={float(hash_feat.norm(dim=-1).mean()):.3f}")
                    
                    # [CHECKLIST A1] sigma_max distribution (critical for diagnosing gate mismatch)
                    if hasattr(gaussians, 'get_scaling'):
                        scaling = gaussians.get_scaling  # [N, 3]
                        sigma_max = scaling.max(dim=-1)[0]  # [N]
                        print(f"  [A1-SIGMA] {percentiles(sigma_max, 'sigma_max')}")
                        
                        # log2(kappa/sigma_max) distribution
                        kappa = hybrid_config.gate_kappa
                        log2_ratio = torch.log2(kappa / (sigma_max + 1e-8))
                        print(f"  [A1-LOG2] {percentiles(log2_ratio, 'log2(κ/σ)')} (κ={kappa:.2f})")
                        
                        # Warn if gate is stuck (all points in same regime)
                        beta_mean = float(beta.mean())
                        if beta_mean < 0.15:
                            print(f"  [A1-WARNING] Beta too low ({beta_mean:.3f}) - VM dominates, Hash blocked!")
                        elif beta_mean > 0.85:
                            print(f"  [A1-WARNING] Beta too high ({beta_mean:.3f}) - Hash dominates, VM unused!")

                # 3) Numeric health check (finite values)
                if DEBUG_CFG.NUMERIC and hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, '_dbg_beta'):
                    vm_finite = torch.isfinite(gaussians._grid._dbg_vm).all().item()
                    hash_finite = torch.isfinite(gaussians._grid._dbg_hash).all().item()
                    beta_finite = torch.isfinite(gaussians._grid._dbg_beta).all().item()
                    gate_finite = torch.isfinite(gaussians._grid._dbg_gate).all().item()
                    all_ok = vm_finite and hash_finite and beta_finite and gate_finite

                    print(f"[DEBUG-NUMERIC {iteration:05d}] "
                          f"vm/hash/beta/gate finite: {vm_finite}/{hash_finite}/{beta_finite}/{gate_finite} -> {'OK' if all_ok else 'FAIL'}")

                # 4) Auxiliary loss values
                if DEBUG_CFG.FUSION:
                    print(f"[DEBUG-AUX {iteration:05d}] vm_aux={float(loss_vm_aux):.6f} orth={float(loss_orth):.6f}")

                # 5) Performance metrics（仅统计打印耗时，主计时仍用 CUDA event）
                if DEBUG_CFG.PERF:
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    step_time_ms = (time.time() - step_start) * 1000
                    print(f"[DEBUG-PERF {iteration:05d}] "
                          f"mem={mem_allocated:.2f}GB/{mem_reserved:.2f}GB "
                          f"step_time={step_time_ms:.1f}ms")
        except Exception:
            # Silently skip debug if any error (don't break training)
            pass

        # ===== 梯度裁剪 + NaN 保险丝（在 step 前执行）=====
        # 聚合两个优化器里的参数
        all_params = []
        for group in gaussians.optimizer.param_groups:
            all_params.extend(group['params'])
        for group in gaussians.optimizer_i.param_groups:
            all_params.extend(group['params'])

        if numeric_config and numeric_config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, numeric_config.grad_clip_norm)

        bad_grad = False
        if numeric_config and numeric_config.nan_guard_enable:
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    bad_grad = True
                    break

        # ====== 关键：正式训练日志（在 step 前打印，避免 0 梯度）=====
        if accel_config.log_every > 0 and iteration % accel_config.log_every == 0:
            enc_type = "hybrid(vm+hash)" if (hybrid_config and hybrid_config.hybrid_enable) else "hash-only"

            # Learning rates
            lr_grid = gaussians.optimizer_i.param_groups[0]['lr']
            # 尝试找到 VM 分组的 lr（若无命名，回退到第二组或相同 lr）
            lr_vm = None
            for pg in gaussians.optimizer_i.param_groups:
                if pg.get('name', '') == 'vm_encoder':
                    lr_vm = pg['lr']
                    break
            if lr_vm is None and len(gaussians.optimizer_i.param_groups) > 1:
                lr_vm = gaussians.optimizer_i.param_groups[1]['lr']

            msg = f"[{iteration:05d}] enc={enc_type} lr={lr_grid:.2e}"
            if lr_vm is not None:
                msg += f" lr_vm={lr_vm:.2e}"

            if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'hash_gain'):
                hg = float(gaussians._grid.hash_gain.item())
                ga = float(gaussians._grid.gate_alpha)
                gtau = float(gaussians._grid.gate_tau)
                msg += f" hash_gain={hg:.3f} gate(α={ga:.1f},τ={gtau:.1f})"

            # 梯度范数（此时仍未 step/zero_grad，数据有效）
            def safe_g(module, attr):
                try:
                    p = getattr(module, attr, None)
                    return float(p.grad.norm()) if (p is not None and p.grad is not None) else 0.0
                except Exception:
                    return 0.0

            if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'vm'):
                g_vm_xy = safe_g(gaussians._grid.vm, 'xy')
                g_vm_mlp = safe_g(gaussians._grid.vm.mlp, 'weight')
                g_proj = safe_g(gaussians._grid.proj_hash, 'weight')
                msg += f" | grad: vm_xy={g_vm_xy:.3e} vm_mlp={g_vm_mlp:.3e} proj_hash={g_proj:.3e}"

            # 损失打印
            msg += f" | loss={float(loss):.4f} pix={float(pixel_loss):.4f} vm_aux={float(loss_vm_aux):.4f} orth={float(loss_orth):.4f}"
            msg += f" loc={float(loss_loc):.4f} feat_reg={float(loss_feat_reg):.4f}"
            if bad_grad:
                msg += " [NaN_SKIP]"

            print(msg)

        # ===== 正式执行 step / 清梯度 =====
        if not bad_grad:
            gaussians.optimizer.step()
            gaussians.optimizer_i.step()
        else:
            print(f"[WARNING] NaN/Inf gradient detected at iteration {iteration}, skipping optimizer step")

        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.optimizer_i.zero_grad(set_to_none=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Num": f"{gaussians.get_xyz.shape[0]:07d}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # [FIX CHECKLIST A5] More aggressive size threshold to prune large Gaussians earlier
                    # Original: 20, New: 16 (20% reduction) to reduce "paint roller" artifacts
                    size_threshold = 16 if iteration > opt.opacity_reset_interval else None
                    # [3DGS BASELINE] Use configurable prune_opacity_threshold (default 0.005)
                    num_pts_before = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold, opt=opt, iteration=iteration)
                    num_pts_after = gaussians.get_xyz.shape[0]
                    if iteration % 1000 == 0:  # Log every 1000 iters to avoid spam
                        print(f"[DENSIFY {iteration}] Points: {num_pts_before} -> {num_pts_after} (Δ{num_pts_after - num_pts_before:+d})")
            else:
                # [3DGS BASELINE] Continue pruning after densification stops
                if iteration % opt.prune_interval == 0:
                    gaussians.mask_prune()

            # [CHECKLIST 8] Enhanced logging: statistics every 1k iterations
            if iteration % 1000 == 0 and hybrid_config and hybrid_config.hybrid_enable:
                try:
                    with torch.no_grad():
                        scaling = gaussians.get_scaling  # [N, 3]
                        if scaling.numel() > 0 and scaling.shape[0] > 0:
                            # m_hf and beta statistics
                            if hasattr(gaussians._grid, '_dbg_gate'):
                                m_hf = gaussians._grid._dbg_gate  # [N]
                                print(f"[STATS {iteration}] m_hf: median={float(torch.median(m_hf)):.4f}, "
                                      f"mean={float(m_hf.mean()):.4f}, std={float(m_hf.std()):.4f}")
                            
                            # log2(sigma_max) distribution
                            sigma_max = scaling.max(dim=-1)[0]  # [N]
                            log2_sigma = torch.log2(sigma_max.clamp(min=1e-8))
                            print(f"[STATS {iteration}] log2(σ_max): median={float(torch.median(log2_sigma)):.3f}, "
                                  f"p25={float(torch.quantile(log2_sigma, 0.25)):.3f}, "
                                  f"p75={float(torch.quantile(log2_sigma, 0.75)):.3f}")
                            
                            # Anisotropy ratio distribution
                            s_min = scaling.min(dim=-1)[0]
                            s_max = scaling.max(dim=-1)[0]
                            aspect_ratio = s_max / (s_min.clamp(min=1e-8))
                            print(f"[STATS {iteration}] aspect_ratio: median={float(torch.median(aspect_ratio)):.2f}, "
                                  f"max={float(aspect_ratio.max()):.2f}, "
                                  f"exceed_r_max({aniso_config.aniso_r_max if aniso_config else 'N/A'})={(aspect_ratio > (aniso_config.aniso_r_max if aniso_config else 999)).sum().item()}")
                            
                            # Point count
                            print(f"[STATS {iteration}] Points: {gaussians.get_xyz.shape[0]}")
                except Exception as e:
                    print(f"[STATS {iteration}] Logging failed: {e}")

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hep = HybridEncoderParams(parser)  # Hybrid encoder config
    vmp = VMauxLossParams(parser)      # VM auxiliary loss config (NEW)
    otp = OrthogonalityParams(parser)  # Orthogonality loss config (NEW)
    nsp = NumericStabilityParams(parser)  # Numeric stability config (NEW)
    wlp = WaveletLossParams(parser)    # Wavelet loss config
    tap = TrainingAccelParams(parser)  # Training acceleration config
    llp = LocalityLossParams(parser)   # Locality loss config (NEW D)
    frp = FeatureRegParams(parser)     # Feature regularization config (GENERALIZATION)
    kap = KappaAdaptiveParams(parser)  # Kappa adaptive config (NEW - CHECKLIST 1)
    acp = AnisotropyConstraintParams(parser)  # Anisotropy constraint config (NEW - CHECKLIST 2)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        hybrid_config=hep.extract(args),   # Pass hybrid encoder params
        vm_aux_config=vmp.extract(args),   # Pass VM aux loss params (NEW)
        orth_config=otp.extract(args),     # Pass orth loss params (NEW)
        numeric_config=nsp.extract(args),  # Pass numeric stability params (NEW)
        wavelet_config=wlp.extract(args),  # Pass wavelet loss params
        accel_config=tap.extract(args),    # Pass training accel params
        loc_config=llp.extract(args),      # Pass locality loss params (NEW D)
        feat_reg_config=frp.extract(args),  # Pass feature reg params (GENERALIZATION)
        kappa_config=kap.extract(args),    # Pass kappa adaptive params (NEW - CHECKLIST 1)
        aniso_config=acp.extract(args)     # Pass anisotropy constraint params (NEW - CHECKLIST 2)
    )

    # All done
    print("\nTraining complete.")
