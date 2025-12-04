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
from utils.loss_utils import l1_loss, ssim, ssim_raw, gradient_loss, compute_pixel_importance
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

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
    
    Args:
        H: Image height
        W: Image width
        alpha: Weight boost at corners (default 0.6 → range [1.0, 1.6])
        device: Target device
    
    Returns:
        weight_map: [H, W] tensor with radial weights
    """
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, 
        dataset.feature_role_split, dataset.geo_resolution, dataset.geo_rank, dataset.geo_channels
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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
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

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            
            # ----------------------------------------------------------------
            # Background-weighted loss: emphasize peripheral regions to give
            # background/edge areas more gradient, encouraging densification
            # there without drastically increasing total Gaussian count.
            # ----------------------------------------------------------------
            H, W = gt_image.shape[1], gt_image.shape[2]
            background_weight_map = build_background_weight_map(H, W, alpha=0.6, device="cuda")
            # Expand to [1, 1, H, W] for broadcasting with [B, C, H, W] images
            weight_map_4d = background_weight_map[None, None, :, :]
            
            # Weighted L1 loss
            rgb_residual = (image - gt_image).abs()
            weighted_rgb_residual = weight_map_4d * rgb_residual
            Ll1 = weighted_rgb_residual.mean()
            
            # Weighted SSIM loss: use per-pixel dissimilarity map
            dssim_map = ssim_raw(image, gt_image)  # [1, 1, H, W]
            ssim_loss_weighted = (weight_map_4d * dssim_map).mean()
            
            pixel_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss_weighted
            
            # Edge loss: controlled by enable_edge_loss flag and start iteration
            # Uses unified edge-aware gradient loss: aligns edges while suppressing flat-region noise
            edge_loss_enabled = getattr(opt, 'enable_edge_loss', False)
            edge_loss_start_iter = getattr(opt, 'edge_loss_start_iter', 0)
            lambda_grad = getattr(opt, 'lambda_grad', 0.0)
            flat_weight = getattr(opt, 'edge_flat_weight', 0.5)  # Weight for flat region suppression
            
            if edge_loss_enabled and iteration >= edge_loss_start_iter and lambda_grad > 0:
                Lgrad, Lgrad_edge, Lgrad_flat = gradient_loss(image, gt_image, flat_weight=flat_weight, return_components=True)
            else:
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
            # Add edge loss only when enabled
            if edge_loss_enabled and iteration >= edge_loss_start_iter and lambda_grad > 0:
                loss = loss + lambda_grad * Lgrad

            # (周期性日志已精简，详细逐项 grad 计算移除以减少频繁的图计算)
        
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
                edge_loss_weight = lambda_grad if edge_loss_enabled else 0.0
                
                # Gaussian statistics
                opacity_vals = gaussians.get_opacity.detach().squeeze()
                if opacity_vals.numel() > 0:
                    opacity_p5 = torch.quantile(opacity_vals, 0.05).item()
                    opacity_p50 = torch.quantile(opacity_vals, 0.50).item()
                    opacity_p95 = torch.quantile(opacity_vals, 0.95).item()
                else:
                    opacity_p5 = opacity_p50 = opacity_p95 = 0.0
                current_sh_degree = gaussians.active_sh_degree
                
                # Capacity info
                max_cap = gaussians.gaussian_capacity_config["max_point_count"]
                cap_pct = gaussian_count / max_cap * 100
                
                # Formatted output
                print("\n" + "=" * 90)
                print(f"  [Iteration {iteration:>5}]  Training Summary")
                print("-" * 90)
                print(f"  Gaussians   │  N = {gaussian_count:,}  │  Capacity = {cap_pct:.1f}%  │  SH Degree = {current_sh_degree}")
                print(f"  Opacity     │  p5 = {opacity_p5:.3f}  │  p50 = {opacity_p50:.3f}  │  p95 = {opacity_p95:.3f}")
                
                # Detail importance statistics (if enabled)
                detail_aware_enabled = getattr(opt, 'enable_detail_aware', True)
                if detail_aware_enabled and hasattr(gaussians, 'detail_importance') and gaussians.detail_importance.numel() > 0:
                    di_stats = gaussians.get_detail_importance_stats()
                    print(f"  Detail Imp  │  p5 = {di_stats['p5']:.3f}  │  p50 = {di_stats['p50']:.3f}  │  p95 = {di_stats['p95']:.3f}")
                
                print("-" * 90)
                print(f"  Loss Total  │  {loss_total_val:.6f}")
                print(f"  Loss RGB    │  L1 = {Ll1.item():.6f}  │  SSIM = {(1.0 - ssim(image, gt_image)).item():.6f}")
                if edge_loss_enabled:
                    print(f"  Loss Edge   │  {edge_loss_value:.6f}  (align = {edge_term_value:.6f}, flat = {flat_term_value:.6f})  │  λ = {edge_loss_weight:.3f}")
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
            # Densification with capacity control (6M hard cap).
            # Delegates to maybe_densify_and_prune which wraps the original
            # densify_and_prune logic without modifying its internals.
            # -----------------------------------------------------------------
            # Keep track of max radii in image-space for pruning (always needed)
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # -----------------------------------------------------------------
            # Detail-aware densification: update per-Gaussian importance scores
            # based on pixel-level edge+error importance from current frame.
            # -----------------------------------------------------------------
            detail_aware_enabled = getattr(opt, 'enable_detail_aware', True)
            if detail_aware_enabled and iteration >= opt.densify_from_iter:
                # Compute pixel importance from GT edge strength and current residual
                power_edge = getattr(opt, 'detail_importance_power_edge', 1.2)
                power_error = getattr(opt, 'detail_importance_power_error', 1.0)
                pixel_importance = compute_pixel_importance(
                    gt_image, image.detach(), 
                    power_edge=power_edge, 
                    power_error=power_error
                )
                # Update per-Gaussian detail_importance via EMA
                gaussians.update_detail_importance(pixel_importance, visibility_filter, radii)
            
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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
