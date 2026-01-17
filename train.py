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
import yaml
from random import randint
from utils.loss_utils import l1_loss, ssim, ssim_raw, compute_edge_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import subprocess
import shutil


def _is_true_env(var_name: str) -> bool:
    return str(os.getenv(var_name, "")).lower() in {"1", "true", "yes", "on"}


def _summarize_param_groups(optimizer) -> list:
    rows = []
    defaults = getattr(optimizer, "defaults", {})
    for idx, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"group_{idx}")
        lr = pg.get("lr", defaults.get("lr", 0.0))
        betas = pg.get("betas", defaults.get("betas", (0.9, 0.999)))
        eps = pg.get("eps", defaults.get("eps", 1e-8))
        wd = pg.get("weight_decay", defaults.get("weight_decay", 0.0))
        num_params = sum(p.numel() for p in pg.get("params", []))
        rows.append({
            "name": name,
            "num_params": num_params,
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": wd,
        })
    return rows

# Enable TF32 tensor cores for faster matmul on Ampere+ GPUs
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def build_background_weight_map(H: int, W: int, alpha: float = 0.6, device: str = "cuda") -> torch.Tensor:
    """
    Build a radial weight map that emphasizes image periphery (edges/corners).
    Center weight ≈ 1.0, corner weight ≈ 1.0 + alpha.
    This encourages densification in background/peripheral regions without
    significantly increasing total Gaussian count.
    
    Uses caching to avoid repeated computation for same resolution.
    
    Args:
        H: Image height
        W: Image width
        alpha: Weight boost at corners (default 0.6 → range [1.0, 1.6])
        device: Target device
    
    Returns:
        weight_map: [H, W] tensor with radial weights
    """
    # Cache lookup to avoid recomputation each iteration
    cache_key = (H, W, alpha, device)
    if not hasattr(build_background_weight_map, '_cache'):
        build_background_weight_map._cache = {}
    if cache_key in build_background_weight_map._cache:
        return build_background_weight_map._cache[cache_key]
    
    # Normalized coordinates in [-1, 1]
    y_coords = torch.linspace(-1.0, 1.0, H, device=device)
    x_coords = torch.linspace(-1.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Radial distance from center, normalized so corners have distance ~1.414
    radial_dist = torch.sqrt(xx ** 2 + yy ** 2)
    # Normalize to [0, 1] range (center=0, corner=1)
    radial_dist_normalized = radial_dist / (2 ** 0.5)
    
    # Linear ramp: center=1.0, corner=1.0+alpha
    weight_map = 1.0 + alpha * radial_dist_normalized
    
    # Store in cache
    build_background_weight_map._cache[cache_key] = weight_map
    return weight_map

try:
    from torch.profiler import (
        profile as torch_profile,
        schedule as profiler_schedule,
        ProfilerActivity,
        tensorboard_trace_handler,
    )
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def save_ablation_config(output_dir, dataset, opt, pipe):
    """Save ablation study configuration to YAML file for experiment tracking."""
    config = {
        'ablation_settings': {
            # Additive ablation controls (A→B→C→D)
            'feature_mod_type': getattr(dataset, 'feature_mod_type', 'film'),
            'densification_mode': getattr(dataset, 'densification_mode', 'mass_aware'),
            'use_edge_loss': getattr(dataset, 'use_edge_loss', True),
        },
        'training_hyperparameters': {
            'iterations': opt.iterations,
            'position_lr_init': opt.position_lr_init,
            'position_lr_final': opt.position_lr_final,
            'lambda_grad': getattr(opt, 'lambda_grad', 0.05),
            'lambda_mask': getattr(opt, 'lambda_mask', 0.01),
            'max_gaussians': getattr(opt, 'max_gaussians', 6_000_000),
            'densify_until_iter': getattr(opt, 'densify_until_iter', 15000),
            'densify_grad_threshold': opt.densify_grad_threshold,
        },
        'dataset_settings': {
            'source_path': dataset.source_path,
            'model_path': dataset.model_path,
            'sh_degree': dataset.sh_degree,
            'hash_size': dataset.hash_size,
            'width': dataset.width,
            'depth': dataset.depth,
            'feature_role_split': dataset.feature_role_split,
            'geo_resolution': dataset.geo_resolution,
            'geo_rank': dataset.geo_rank,
            'geo_channels': dataset.geo_channels,
        },
        'rendering_settings': {
            'white_background': dataset.white_background,
            'convert_SHs_python': pipe.convert_SHs_python,
            'compute_cov3D_python': pipe.compute_cov3D_python,
        }
    }
    
    config_path = os.path.join(output_dir, 'ablation_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[Config] Saved ablation configuration to: {config_path}\n")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # Debug flag for full ablation gap diagnostics (read-only logging)
    debug_full_gap = pipe.debug_full_gap or os.getenv("DEBUG_FULL_GAP", "0") in {"1", "true", "True"}
    
    # ========================================================================
    # Additive ablation controls (A→B→C→D)
    # A: Concat (baseline) - naive feature fusion
    # B: +FiLM - add FiLM modulation (geometry guides appearance)
    # C: +MassAware - add mass-aware densification
    # D: +EdgeLoss - add edge-aware gradient loss (full GlowGS)
    # ========================================================================
    feature_mod_type = getattr(dataset, 'feature_mod_type', 'film')  # 'concat' or 'film'
    densification_mode = getattr(dataset, 'densification_mode', 'mass_aware')  # 'standard' or 'mass_aware'
    use_edge_loss = getattr(dataset, 'use_edge_loss', True)
    
    # Always use hybrid encoder (GlowGS core architecture)
    encoder_variant = 'hybrid'
    
    # Map edge loss setting
    edge_loss_mode = 'sobel_weighted' if use_edge_loss else 'none'
    
    # Map densification mode to internal strategy name
    densify_strategy = 'feature_weighted' if densification_mode == 'mass_aware' else 'original_3dgs'
    
    # Store feature_mod_type in dataset for encoder to use
    dataset.feature_mod_type = feature_mod_type
    
    # Store edge_loss_mode in opt for consistent access
    opt.edge_loss_mode = edge_loss_mode
    
    # Save ablation study configuration for experiment tracking
    save_ablation_config(dataset.model_path, dataset, opt, pipe)
    
    gaussians = GaussianModel(
        dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, 
        dataset.feature_role_split, dataset.geo_resolution, dataset.geo_rank, dataset.geo_channels,
        encoder_variant=encoder_variant,
        densify_strategy=densify_strategy,
        feature_mod_type=feature_mod_type,
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize AMP (Automatic Mixed Precision) if enabled
    use_amp = getattr(opt, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[AMP] Automatic Mixed Precision enabled - faster training with lower memory usage")

    # Debug counters for densify/prune (read-only)
    densify_events = 0
    prune_events = 0

    # --------------------------------------------------------------------
    # Debug Full Gap: one-shot configuration snapshots (read-only)
    # --------------------------------------------------------------------
    if debug_full_gap:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        sample_cam = train_cams[0] if len(train_cams) > 0 else None
        res_info = f"{sample_cam.image_width}x{sample_cam.image_height}" if sample_cam else "N/A"
        eval_res_strategy = "original" if getattr(dataset, "resolution", -1) == -1 else f"scaled /r={dataset.resolution}"
        eval_split = "LLFF hold every 8" if getattr(dataset, "eval", False) else "train-only"
        color_space = "linear RGB (no gamma)"  # metrics use tensors directly
        clamp_mode = "render outputs clamped to [0,1] in eval"  # see training_report
        bg_mode = "white" if dataset.white_background else ("random" if opt.random_background else "black")
        # Training schedule snapshot
        densify_start = getattr(opt, 'densify_from_iter', 500)
        densify_end = getattr(opt, 'densify_until_iter', 15000)
        densify_interval = getattr(opt, 'densification_interval', 100)
        opacity_reset_iter = getattr(opt, 'opacity_reset_interval', 3000)
        sh_sched = f"+1 degree every 1000 iters up to {dataset.sh_degree}"
        # Loss weights (effective)
        loss_weights = {
            "lambda_dssim": opt.lambda_dssim,
            "lambda_mask": getattr(opt, 'lambda_mask', 0.0),
            "lambda_sh_mask": getattr(opt, 'lambda_sh_mask', 0.0),
            "lambda_grad": getattr(opt, 'lambda_grad', 0.0),
        }
        # Point budget
        cap_cfg = gaussians.gaussian_capacity_config
        point_budget = {
            "max_gaussians": cap_cfg.get("max_point_count", getattr(opt, 'max_gaussians', 0)),
            "densify_until": cap_cfg.get("densify_until_iter", densify_end),
            "prune_interval": cap_cfg.get("prune_interval", getattr(opt, 'prune_interval', 1000)),
            "min_opacity": getattr(opt, 'min_opacity', 0.005),
            "mask_prune_threshold": getattr(opt, 'mask_prune_threshold', 0.01),
            "densify_grad_threshold": getattr(opt, 'densify_grad_threshold', 0.0002),
            "init_gaussians": gaussians.get_xyz.shape[0],
        }
        # Optimizer groups snapshot
        def _summarize_optimizer(opt_obj, title):
            rows = []
            for i, pg in enumerate(opt_obj.param_groups):
                name = pg.get('name', f'group_{i}')
                lr = pg.get('lr', 0.0)
                betas = pg.get('betas', (0.9, 0.999))
                eps = pg.get('eps', 1e-8)
                wd = pg.get('weight_decay', 0.0)
                n_params = sum(p.numel() for p in pg['params'])
                rows.append((name, n_params, lr, betas, eps, wd))
            print(f"[DEBUG_FULL_GAP] Optimizer Snapshot - {title}")
            print("name | #params | lr | betas | eps | weight_decay")
            for r in rows:
                print(f"  {r[0]:12s} | {r[1]:8d} | {r[2]:.6g} | ({r[3][0]:.2f},{r[3][1]:.2f}) | {r[4]:.1e} | {r[5]:.2g}")

        print("\n[DEBUG_FULL_GAP] Eval Snapshot")
        print(f"  color_space       : {color_space}")
        print(f"  clamp_mode        : {clamp_mode}")
        print(f"  background        : {bg_mode}")
        print(f"  eval_resolution   : {eval_res_strategy} (sample {res_info})")
        print(f"  test_views        : {len(test_cams)} from {eval_split}")

        print("[DEBUG_FULL_GAP] Train Snapshot")
        print(f"  total_iters       : {opt.iterations}")
        print(f"  test_every        : {testing_iterations}")
        print(f"  densify start/end : {densify_start}/{densify_end} (interval {densify_interval})")
        print(f"  opacity_reset     : {opacity_reset_iter}")
        print(f"  sh_schedule       : {sh_sched}")
        for k,v in loss_weights.items():
            print(f"  {k:15s}: {v}")

        print("[DEBUG_FULL_GAP] Point Budget Snapshot")
        for k,v in point_budget.items():
            print(f"  {k:20s}: {v}")

        print("[DEBUG_FULL_GAP] Optimizer Param Groups (explicit)")
        _summarize_optimizer(gaussians.optimizer, "explicit")
        print("[DEBUG_FULL_GAP] Optimizer Param Groups (encoder/MLP)")
        _summarize_optimizer(gaussians.optimizer_i, "implicit")

        print("[DEBUG_FULL_GAP] RNG / Sampling Snapshot")
        print(f"  torch.initial_seed : {torch.initial_seed()}")
        print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  cudnn.benchmark    : {torch.backends.cudnn.benchmark}")
        print(f"  AMP enabled        : {use_amp}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Print ablation study configuration summary (ECCV A→B→C→D)
    print("\n" + "=" * 90)
    print("  ABLATION STUDY CONFIGURATION")
    print("-" * 90)
    print(f"  [GlowGS] feature_mod_type      : {feature_mod_type}")
    print(f"  [GlowGS] densification_mode    : {densification_mode}")
    print(f"  [GlowGS] use_edge_loss         : {use_edge_loss}")
    print("-" * 90)
    
    # Derive variant label (Additive ablation study A→B→C→D)
    if feature_mod_type == 'concat' and densification_mode == 'standard' and not use_edge_loss:
        variant_label = "A: Concat (Baseline)"
    elif feature_mod_type == 'film' and densification_mode == 'standard' and not use_edge_loss:
        variant_label = "B: +FiLM Modulation"
    elif feature_mod_type == 'film' and densification_mode == 'mass_aware' and not use_edge_loss:
        variant_label = "C: +Mass-Aware Densify"
    elif feature_mod_type == 'film' and densification_mode == 'mass_aware' and use_edge_loss:
        variant_label = "D: Full GlowGS"
    else:
        variant_label = "Custom"
    
    print(f"  Variant              │  {variant_label}")
    print(f"  Max Gaussians        │  {getattr(opt, 'max_gaussians', 6_000_000):,}")
    print(f"  Iterations           │  {opt.iterations:,}")
    print("=" * 90 + "\n")
    
    # ========================================================================
    # EQUIVALENCE DEBUG MODE: detailed configuration snapshot for repro
    # ========================================================================
    if pipe.debug_equiv:
        print("\n" + "#" * 90)
        print("  🔍 EQUIVALENCE DEBUG MODE ENABLED")
        print("#" * 90)
        
        # Dataset config
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        if len(train_cams) > 0:
            sample_cam = train_cams[0]
            resolution_info = f"{sample_cam.image_width}x{sample_cam.image_height}"
        else:
            resolution_info = "N/A"
        
        print("\n[Dataset]")
        print(f"  Train views      : {len(train_cams)}")
        print(f"  Test views       : {len(test_cams)}")
        print(f"  Resolution       : {resolution_info}")
        print(f"  White background : {dataset.white_background}")
        print(f"  Random bg        : {opt.random_background}")
        print(f"  SH degree        : {dataset.sh_degree}")
        
        # Schedule config
        print("\n[Training Schedule]")
        print(f"  Total iters         : {opt.iterations}")
        print(f"  Densify start       : {getattr(opt, 'densify_from_iter', 500)}")
        print(f"  Densify until       : {getattr(opt, 'densify_until_iter', 15000)}")
        print(f"  Densify interval    : {getattr(opt, 'densification_interval', 100)}")
        print(f"  Densify threshold   : {opt.densify_grad_threshold}")
        print(f"  Opacity reset       : {getattr(opt, 'opacity_reset_interval', 3000)}")
        print(f"  SH degree schedule  : every 1000 iters (max={dataset.sh_degree})")
        
        # Randomness
        print("\n[Randomness]")
        print(f"  Torch seed       : {torch.initial_seed()}")
        print(f"  CUDA deterministic : {torch.backends.cudnn.deterministic}")
        
        # Optimizer param groups
        print("\n[Optimizer: Explicit Params]")
        for i, pg in enumerate(gaussians.optimizer.param_groups):
            pg_name = pg.get('name', f'group_{i}')
            pg_lr = pg['lr']
            pg_params = sum(p.numel() for p in pg['params'])
            print(f"  {pg_name:15s} │ lr={pg_lr:.6f} │ params={pg_params:>8,d}")
        
        print("\n[Optimizer: Implicit Params (MLP/Grid)]")
        for i, pg in enumerate(gaussians.optimizer_i.param_groups):
            pg_name = pg.get('name', f'group_{i}')
            pg_lr = pg['lr']
            pg_params = sum(p.numel() for p in pg['params'])
            print(f"  {pg_name:15s} │ lr={pg_lr:.6f} │ params={pg_params:>8,d}")
        
        # Loss config
        print("\n[Loss Weights]")
        print(f"  lambda_dssim     : {opt.lambda_dssim}")
        print(f"  lambda_mask      : {getattr(opt, 'lambda_mask', 0.004)}")
        print(f"  lambda_sh_mask   : {getattr(opt, 'lambda_sh_mask', 0.0001)}")
        print(f"  lambda_grad      : {getattr(opt, 'lambda_grad', 0.05)} (edge loss, starts iter {getattr(opt, 'edge_loss_start_iter', 5000)})")
        
        # AMP
        print("\n[Training Tech]")
        print(f"  AMP (mixed precision) : {use_amp}")
        print(f"  Detail-aware densify  : {getattr(opt, 'enable_detail_aware', True)}")
        
        print("\n" + "#" * 90 + "\n")

    # ====================================================================
    # DEBUG_FULL_GAP: one-shot snapshots (read-only, no behavior changes)
    # ====================================================================
    if debug_full_gap:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        eval_color_space = "linear tensor [0,1] (no sRGB gamma)"
        clamp_eval = "clamp to [0,1] in eval (render/report)"
        bg_mode = "white" if dataset.white_background else ("random-bg" if opt.random_background else "black")
        eval_res_flag = getattr(dataset, "resolution", getattr(dataset, "_resolution", -1))
        test_source = "llffhold=8 split" if dataset.eval else "train-only (no held-out)"
        test_source = f"{test_source}, test_views={len(test_cams)}"

        print("[DEBUG_FULL_GAP] Eval Snapshot")
        print("setting | value")
        print(f"color_space | {eval_color_space}")
        print(f"render_clamp | {clamp_eval}")
        print(f"background | {bg_mode}")
        print(f"eval_resolution_flag | {eval_res_flag}")
        print(f"test_views_source | {test_source}")

        print("[DEBUG_FULL_GAP] Train Snapshot")
        print("setting | value")
        print(f"total_iters | {opt.iterations}")
        print(f"test_iterations | {testing_iterations}")
        print(f"densify_start/end/interval | {opt.densify_from_iter}/{opt.densify_until_iter}/{opt.densification_interval}")
        print(f"opacity_reset_interval | {opt.opacity_reset_interval}")
        print(f"sh_degree_schedule | +1 per 1000 iters up to {dataset.sh_degree}")
        print(f"loss_weights | dssim={opt.lambda_dssim} mask={getattr(opt,'lambda_mask',0.0)} sh_mask={getattr(opt,'lambda_sh_mask',0.0)} grad={getattr(opt,'lambda_grad',0.0)} flat_w={getattr(opt,'edge_flat_weight',0.0)}")

        print("[DEBUG_FULL_GAP] Point Budget Snapshot")
        print("setting | value")
        print(f"max_gaussians | {getattr(opt, 'max_gaussians', gaussians.gaussian_capacity_config['max_point_count'])}")
        print(f"initial_gaussians | {gaussians.get_xyz.shape[0]:,}")
        print(f"densify_threshold | {opt.densify_grad_threshold}")
        print(f"min_opacity | {getattr(opt,'min_opacity',0.005)}")
        print(f"mask_prune_threshold | {getattr(opt,'mask_prune_threshold',0.01)}")

        print("[DEBUG_FULL_GAP] Optimizer Snapshot (explicit)")
        print("name | num_params | lr | betas | eps | weight_decay")
        for row in _summarize_param_groups(gaussians.optimizer):
            print(f"{row['name']} | {row['num_params']:,} | {row['lr']:.6f} | {row['betas']} | {row['eps']:.1e} | {row['weight_decay']}")
        print("[DEBUG_FULL_GAP] Optimizer Snapshot (implicit encoder/MLP)")
        print("name | num_params | lr | betas | eps | weight_decay")
        for row in _summarize_param_groups(gaussians.optimizer_i):
            print(f"{row['name']} | {row['num_params']:,} | {row['lr']:.6f} | {row['betas']} | {row['eps']:.1e} | {row['weight_decay']}")

        print("[DEBUG_FULL_GAP] RNG/Sampling Snapshot")
        print("setting | value")
        print(f"torch_seed | {torch.initial_seed()}")
        print(f"numpy_seed | 0 (safe_state)")
        print(f"python_seed | 0 (safe_state)")
        print(f"cudnn_deterministic | {torch.backends.cudnn.deterministic}")
        print(f"cudnn_benchmark | {torch.backends.cudnn.benchmark}")
        print(f"amp_gradscaler | {use_amp}")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    # Counters for debug summaries
    densify_events = 0
    prune_events = 0
    
    # Equivalence debug: track first N iters for reproducibility check
    debug_n_iters = 5 if pipe.debug_equiv else 0
    if pipe.debug_equiv and first_iter == 1:
        print("\n[Debug] Tracking first 5 iterations in detail...\n")
        print(f"{'Iter':>5} | {'CamID':>6} | {'N_Gauss':>8} | {'Loss':>10} | {'RGB Mean':>10} | {'RGB Std':>10}")
        print("-" * 75)
    
    profiler = None
    profiler_logdir = None
    profile_iter_count = 0
    if getattr(opt, "profile_iters", 0) > 0:
        if not PROFILER_AVAILABLE:
            print("[WARN] torch.profiler not available, profiling disabled.")
        else:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            prof_schedule = profiler_schedule(
                wait=getattr(opt, "profile_wait", 2),
                warmup=getattr(opt, "profile_warmup", 2),
                active=getattr(opt, "profile_active", 4),
                repeat=1
            )
            profiler_logdir = getattr(opt, "profile_logdir", "") or os.path.join(scene.model_path, "profiler")
            os.makedirs(profiler_logdir, exist_ok=True)
            profiler = torch_profile(
                activities=activities,
                schedule=prof_schedule,
                on_trace_ready=tensorboard_trace_handler(profiler_logdir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Use AMP autocast for forward pass
        with torch.cuda.amp.autocast(enabled=use_amp):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # Debug equiv: log first N iters in detail
            if iteration <= debug_n_iters:
                cam_id = viewpoint_cam.uid
                n_gauss = gaussians.get_xyz.shape[0]
                rgb_mean = image.mean().item()
                rgb_std = image.std().item()

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            
            # ----------------------------------------------------------------
            # Standard pixel-wise loss: treat all regions equally for optimal PSNR
            # ----------------------------------------------------------------
            Ll1 = l1_loss(image, gt_image)
            ssim_loss_weighted = 1.0 - ssim(image, gt_image)
            pixel_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss_weighted
            
            # Edge loss: unified interface for ablation studies
            # Mode can be "none" | "sobel_basic" | "laplacian_weighted" (paper default)
            edge_loss_mode = getattr(opt, 'edge_loss_mode', 'laplacian_weighted')
            edge_loss_start_iter = getattr(opt, 'edge_loss_start_iter', 0)
            lambda_grad = getattr(opt, 'lambda_grad', 0.05)
            flat_weight = getattr(opt, 'edge_flat_weight', 0.5)
            
            if iteration >= edge_loss_start_iter:
                Lgrad, Lgrad_edge, Lgrad_flat = compute_edge_loss(
                    image, gt_image, 
                    mode=edge_loss_mode,
                    lambda_edge=lambda_grad,
                    flat_weight=flat_weight,
                    return_components=True
                )
            else:
                # Before start_iter: no edge loss
                Lgrad = torch.tensor(0.0, device=image.device)
                Lgrad_edge = torch.tensor(0.0, device=image.device)
                Lgrad_flat = torch.tensor(0.0, device=image.device)
            
            mask_loss = torch.mean(gaussians.get_mask)
            sh_mask_loss = 0.0
            if iteration > opt.densify_until_iter:
                for degree in range(1, gaussians.active_sh_degree + 1):
                    lambda_degree = (2 * degree + 1) / ((gaussians.max_sh_degree + 1) ** 2 - 1)
                    # 改进：对高阶SH项使用更小的权重，避免过度抑制view-dependent效果
                    weight = 1.0 if degree <= 2 else 0.3  # 3阶及以上权重降低
                    sh_mask_loss += weight * lambda_degree * torch.mean(gaussians.get_sh_mask[..., degree - 1])

            loss = pixel_loss + opt.lambda_mask * mask_loss + opt.lambda_sh_mask * sh_mask_loss
            # Add edge loss (already weighted by lambda_grad in compute_edge_loss)
            loss = loss + Lgrad

            # (周期性日志已精简，详细逐项 grad 计算移除以减少频繁的图计算)
        
        # Debug equiv: print first N iters
        if iteration <= debug_n_iters:
            print(f"{iteration:>5} | {cam_id:>6} | {n_gauss:>8,} | {loss.item():>10.6f} | {rgb_mean:>10.6f} | {rgb_std:>10.6f}")
        
        # (已移除重复的逐项 debug 打印，保留更简洁的周期性报告)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # Unified detailed logging every 1000 iterations
            if iteration % 1000 == 0:
                pixel_loss_value = pixel_loss.item()
                loss_mask_val = mask_loss.item()
                loss_sh_mask_val = sh_mask_loss.item() if isinstance(sh_mask_loss, torch.Tensor) else sh_mask_loss
                loss_total_val = loss.item()
                gaussian_count = gaussians.get_xyz.shape[0]
                edge_loss_value = Lgrad.item() if isinstance(Lgrad, torch.Tensor) else 0.0
                edge_term_value = Lgrad_edge.item() if isinstance(Lgrad_edge, torch.Tensor) else 0.0
                flat_term_value = Lgrad_flat.item() if isinstance(Lgrad_flat, torch.Tensor) else 0.0
                
                # Edge loss weight
                edge_loss_mode = getattr(opt, 'edge_loss_mode', 'laplacian_weighted')
                edge_loss_weight = lambda_grad if edge_loss_mode != "none" else 0.0
                
                # Gaussian statistics
                opacity_vals = gaussians.get_opacity.detach().squeeze()
                if opacity_vals.numel() > 0:
                    opacity_p5 = torch.quantile(opacity_vals, 0.05).item()
                    opacity_p50 = torch.quantile(opacity_vals, 0.50).item()
                    opacity_p95 = torch.quantile(opacity_vals, 0.95).item()
                    opacity_mean = opacity_vals.mean().item()
                else:
                    opacity_p5 = opacity_p50 = opacity_p95 = opacity_mean = 0.0
                current_sh_degree = gaussians.active_sh_degree
                if debug_full_gap:
                    scaling_vals = gaussians.get_scaling.detach()
                    if scaling_vals.numel() > 0:
                        scale_max = scaling_vals.max(dim=1).values
                        scale_p90 = torch.quantile(scale_max, 0.90).item()
                    else:
                        scale_p90 = 0.0
                    train_psnr_val = psnr(torch.clamp(image, 0.0, 1.0), torch.clamp(gt_image, 0.0, 1.0)).mean().item()
                    print(f"[DFG] iter={iteration:05d} psnr={train_psnr_val:.3f} N={gaussian_count:,} sh={current_sh_degree} densify={densify_events} prune={prune_events} mean_opac={opacity_mean:.4f} scale_p90={scale_p90:.4f}")
                
                # Capacity info
                max_cap = gaussians.gaussian_capacity_config["max_point_count"]
                cap_pct = gaussian_count / max_cap * 100
                
                # Formatted output
                print("\n" + "=" * 90)
                print(f"  [Iteration {iteration:>5}]  Training Summary")
                print("-" * 90)
                print(f"  Gaussians   │  N = {gaussian_count:,}  │  Capacity = {cap_pct:.1f}%  │  SH Degree = {current_sh_degree}")
                print(f"  Opacity     │  p5 = {opacity_p5:.3f}  │  p50 = {opacity_p50:.3f}  │  p95 = {opacity_p95:.3f}")
                
                print("-" * 90)
                print(f"  Loss Total  │  {loss_total_val:.6f}")
                print(f"  Loss RGB    │  L1 = {Ll1.item():.6f}  │  SSIM = {(1.0 - ssim(image, gt_image)).item():.6f}")
                if edge_loss_mode != "none":
                    print(f"  Loss Edge   │  {edge_loss_value:.6f}  (align = {edge_term_value:.6f}, flat = {flat_term_value:.6f})  │  λ = {edge_loss_weight:.3f}  │  mode = {edge_loss_mode}")
                print("=" * 90 + "\n")
            
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

            # -----------------------------------------------------------------
            # Densification with capacity control.
            # -----------------------------------------------------------------
            # Keep track of max radii in image-space for pruning (always needed)
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # Pass opacity and radii for Mass-Aware Gradient Weighting
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
                                              opacity=gaussians.get_opacity, radii=radii)
            
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                torch.cuda.empty_cache()
                n_before = gaussians.get_xyz.shape[0]
                
                # Use configured min_opacity (LocoGS default: 0.005)
                min_opacity = getattr(opt, 'min_opacity', 0.005)
                mask_prune_threshold = getattr(opt, 'mask_prune_threshold', 0.005)
                
                # Call capacity-controlled wrapper instead of direct densify_and_prune
                gaussians.maybe_densify_and_prune(
                    iteration=iteration,
                    max_grad=opt.densify_grad_threshold,
                    min_opacity=min_opacity,
                    extent=scene.cameras_extent,
                    max_screen_size=size_threshold,
                    mask_prune_threshold=mask_prune_threshold,
                )
                
                n_after = gaussians.get_xyz.shape[0]
                if debug_full_gap:
                    densify_events += int(n_after > n_before)
                    prune_events += int(n_after < n_before)
                torch.cuda.empty_cache()
            
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Optimizer step with AMP gradient scaling
            if iteration < opt.iterations:
                # Unscale gradients and step optimizers
                scaler.step(gaussians.optimizer)
                scaler.step(gaussians.optimizer_i)
                scaler.update()
                
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_i.zero_grad(set_to_none = True)

        if profiler:
            profiler.step()
            profile_iter_count += 1
            if profile_iter_count >= getattr(opt, "profile_iters", 0):
                profiler.stop()
                print(f"[Profiler] Trace captured. Logs stored at {profiler_logdir}")
                profiler = None

    if profiler:
        profiler.stop()
        print(f"[Profiler] Trace captured. Logs stored at {profiler_logdir}")

    # ========================================================================
    # Final Evaluation & Stats Export (for ECCV ablation study)
    # ========================================================================
    print("\n" + "=" * 90)
    print("  FINAL EVALUATION & STATS EXPORT")
    print("=" * 90)
    
    # Import required modules for final evaluation
    from lpipsPyTorch import lpips
    import json
    import time
    
    with torch.no_grad():
        # Get test cameras for final evaluation
        test_cams = scene.getTestCameras()
        
        if len(test_cams) > 0:
            # Final metrics computation
            final_psnrs = []
            final_ssims = []
            final_lpipss = []
            
            # Precompute attributes for fast rendering
            gaussians.precompute_attributes()
            
            # Render all test views and compute metrics
            for viewpoint in test_cams:
                image = torch.clamp(render(viewpoint, gaussians, pipe, background)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                
                final_psnrs.append(psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item())
                final_ssims.append(ssim_raw(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item())
                final_lpipss.append(lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg').item())
            
            avg_psnr = sum(final_psnrs) / len(final_psnrs)
            avg_ssim = sum(final_ssims) / len(final_ssims)
            avg_lpips = sum(final_lpipss) / len(final_lpipss)
            
            # FPS measurement via external script (fps_test.py)
            print("Measuring FPS via fps_test.py...")
            fps = 0.0
            try:
                fps_cmd = [sys.executable, "fps_test.py", "-m", dataset.model_path, "--iteration", str(opt.iterations), "--quiet", "--skip_train", "--skip_test"]
                result = subprocess.run(fps_cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode != 0:
                     print(f"fps_test.py failed with return code {result.returncode}")
                     print(result.stderr)
                else:
                    import re
                    match = re.search(r"Average FPS\s*:\s*([\d\.]+)", result.stdout)
                    if match:
                        fps = float(match.group(1))
                        print(f"  Parsed FPS: {fps}")
                    else:
                        print("  Could not parse FPS from output.")
                        print(result.stdout)
            except Exception as e:
                print(f"Error running fps_test.py: {e}")
            
            # Model size calculation (GPCC compressed)
            print("Computing compressed model size (GPCC)...")
            from utils.gpcc_utils import encode_xyz
            
            size_mb = 0.0
            try:
                temp_dir = os.path.join(dataset.model_path, "gpcc_temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                encode_xyz(xyz, temp_dir, show=False)
                
                bin_path = os.path.join(temp_dir, "xyz.bin")
                if os.path.exists(bin_path):
                    size_mb = os.path.getsize(bin_path) / (1024 * 1024)
                    print(f"  Compressed Size: {size_mb:.2f} MB")
                else:
                    print("  Compression output xyz.bin not found.")
                
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error computing compressed size: {e}")
                # Fallback to PLY size if compression fails
                ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{opt.iterations}", "point_cloud.ply")
                if os.path.exists(ply_path):
                    size_mb = os.path.getsize(ply_path) / (1024 * 1024)
            
            # Extract scene name from source_path
            scene_name = os.path.basename(dataset.source_path.rstrip('/'))
            
            # Determine method name from ECCV ablation config (A→B→C→D)
            if feature_mod_type == 'concat' and densification_mode == 'standard' and not use_edge_loss:
                method_name = "A_Concat"
            elif feature_mod_type == 'film' and densification_mode == 'standard' and not use_edge_loss:
                method_name = "B_FiLM"
            elif feature_mod_type == 'film' and densification_mode == 'mass_aware' and not use_edge_loss:
                method_name = "C_MassAware"
            elif feature_mod_type == 'film' and densification_mode == 'mass_aware' and use_edge_loss:
                method_name = "D_Full"
            else:
                method_name = "Custom"
            
            # Create stats dictionary
            stats = {
                "scene": scene_name,
                "method": method_name,
                "psnr": round(avg_psnr, 4),
                "ssim": round(avg_ssim, 4),
                "lpips": round(avg_lpips, 4),
                "size_mb": round(size_mb, 2),
                "fps": round(fps, 1),
                "num_gaussians": gaussians.get_xyz.shape[0],
                "config": {
                    "feature_mod_type": feature_mod_type,
                    "densification_mode": densification_mode,
                    "use_edge_loss": use_edge_loss,
                }
            }
            
            # Save stats.json
            stats_path = os.path.join(dataset.model_path, "stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\n  Scene: {scene_name}")
            print(f"  Method: {method_name}")
            print(f"  PSNR: {avg_psnr:.4f}  |  SSIM: {avg_ssim:.4f}  |  LPIPS: {avg_lpips:.4f}")
            print(f"  Size: {size_mb:.2f} MB  |  FPS: {fps:.1f}")
            print(f"  Gaussians: {gaussians.get_xyz.shape[0]:,}")
            print(f"\n  Stats saved to: {stats_path}")
        else:
            print("  [WARN] No test cameras available for final evaluation")
    
    print("=" * 90 + "\n")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    sys.stderr.write(f"Loading scene for training: {args.model_path}...\n")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
