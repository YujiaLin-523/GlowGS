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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# [HYBRID] Enable TF32 for faster matmul on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# WAVELET LOSS (Optional, image-domain high-frequency emphasis)
# ============================================================================
def compute_wavelet_loss(pred_img, gt_img, wavelet_config):
    """
    Optional wavelet decomposition loss for high-frequency details.
    
    Decomposes images into subbands (LL, LH, HL, HH) via DWT and applies
    weighted L1 loss. Emphasizes high-frequency reconstruction without
    modifying renderer or encoder internals.
    
    Args:
        pred_img: [H, W, 3] predicted image tensor
        gt_img: [H, W, 3] ground truth image tensor
        wavelet_config: WaveletLossParams object
    
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
    
    # Weighted combination
    total_loss = wavelet_config.wavelet_lambda_ll * ll_loss + wavelet_config.wavelet_lambda_h * hf_loss
    return total_loss


# ============================================================================
# HELPER: Gradient Norm Logging
# ============================================================================
def get_grad_norm(param):
    """Safely extract gradient norm from a parameter."""
    if param is not None and param.grad is not None:
        return float(param.grad.norm().detach())
    return 0.0

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, hybrid_config=None, wavelet_config=None, accel_config=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, hybrid_config=hybrid_config)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # [HYBRID] Setup AMP scaler for mixed precision training
    amp_dtype = torch.float16 if accel_config.amp_dtype == 'float16' else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=accel_config.amp_enable)
    
    # [HYBRID] Optional torch.compile for encoder (experimental)
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
    
    # [HYBRID] Print configuration at training start
    enc_type = "hybrid(vm+hash)" if (hybrid_config and hybrid_config.hybrid_enable) else "hash-only"
    print(f"[TRAINING] Encoder: {enc_type}")
    if hybrid_config and hybrid_config.hybrid_enable:
        print(f"[TRAINING] VM config: rank={hybrid_config.vm_rank}, res={hybrid_config.vm_plane_res}, out_dim={hybrid_config.vm_out_dim}")
    print(f"[TRAINING] AMP: {accel_config.amp_enable} ({accel_config.amp_dtype}), GradClip: {accel_config.grad_clip_norm}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

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

        # [HYBRID] Wrap forward pass in AMP autocast
        with torch.cuda.amp.autocast(enabled=accel_config.amp_enable, dtype=amp_dtype):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss computation
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            pixel_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # [HYBRID] Optional wavelet loss (image domain)
            wavelet_loss = compute_wavelet_loss(image.permute(1, 2, 0), gt_image.permute(1, 2, 0), wavelet_config)
            
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

            # [HYBRID] Backward with AMP scaling
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            
            # [FIX] Gradient amplification for VM encoder to combat vanishing gradients
            if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'vm'):
                vm_grad_scale = 3.0  # Amplify VM gradients by 3x
                for param in gaussians._grid.vm.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(vm_grad_scale)
            
            # Gradient clipping
            if accel_config.grad_clip_norm > 0:
                scaler.unscale_(gaussians.optimizer)
                scaler.unscale_(gaussians.optimizer_i)
                torch.nn.utils.clip_grad_norm_(gaussians.parameters(), accel_config.grad_clip_norm)
            
            # [HYBRID] Debug logging AFTER backward, BEFORE optimizer step (to capture gradients)
            if iteration % 500 == 0:
                # Get current learning rates
                lr_xyz = gaussians.optimizer.param_groups[0]['lr']
                lr_grid = gaussians.optimizer_i.param_groups[0]['lr']
                
                msg = f"[ITER {iteration}] enc={enc_type} loss={float(loss):.4f} lr_xyz={lr_xyz:.2e} lr_grid={lr_grid:.2e}"
                
                # Gradient norms for hybrid encoder components (AFTER backward, BEFORE zero_grad)
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'proj_hash'):
                    try:
                        gnorm_proj = get_grad_norm(gaussians._grid.proj_hash.weight)
                        msg += f" | grad_proj_hash={gnorm_proj:.3f}"
                    except:
                        pass
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'vm'):
                    try:
                        # VM plane gradients
                        gnorm_vm_xy = get_grad_norm(gaussians._grid.vm.xy)
                        gnorm_vm_mlp = get_grad_norm(gaussians._grid.vm.mlp.weight)
                        msg += f" | grad_vm_plane={gnorm_vm_xy:.3f} grad_vm_mlp={gnorm_vm_mlp:.3f}"
                    except:
                        pass
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'hash'):
                    try:
                        # Hash encoder gradient (tcnn internal params)
                        gnorm_hash = get_grad_norm(gaussians._grid.hash.params)
                        msg += f" | grad_hash={gnorm_hash:.3f}"
                    except:
                        pass
                
                # Network heads gradients for comparison
                try:
                    gnorm_opacity = get_grad_norm(gaussians._opacity_head.params)
                    gnorm_frest = get_grad_norm(gaussians._features_rest_head.params)
                    msg += f" | grad_opacity={gnorm_opacity:.3f} grad_frest={gnorm_frest:.3f}"
                except:
                    pass
                
                # Feature scale value (learnable amplification factor)
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'feature_scale'):
                    try:
                        feat_scale = float(gaussians._grid.feature_scale.item())
                        msg += f" | feat_scale={feat_scale:.2f}"
                    except:
                        pass
                
                print(msg)
            
            scaler.step(gaussians.optimizer)
            scaler.step(gaussians.optimizer_i)
            scaler.update()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.optimizer_i.zero_grad(set_to_none=True)
        else:
            # [ORIG] Standard backward without AMP
            loss.backward()
            
            # [FIX] Gradient amplification for VM encoder to combat vanishing gradients
            if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'vm'):
                vm_grad_scale = 3.0  # Amplify VM gradients by 3x
                for param in gaussians._grid.vm.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(vm_grad_scale)
            
            if accel_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(gaussians.parameters(), accel_config.grad_clip_norm)
            
            # [HYBRID] Debug logging AFTER backward, BEFORE optimizer step (to capture gradients)
            if iteration % 1000 == 0:
                # Get current learning rates
                lr_xyz = gaussians.optimizer.param_groups[0]['lr']
                lr_grid = gaussians.optimizer_i.param_groups[0]['lr']
                
                msg = f"[ITER {iteration}] enc={enc_type} loss={float(loss):.4f} lr_xyz={lr_xyz:.2e} lr_grid={lr_grid:.2e}"
                
                # Gradient norms for hybrid encoder components (AFTER backward, BEFORE zero_grad)
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'proj_hash'):
                    try:
                        gnorm_proj = get_grad_norm(gaussians._grid.proj_hash.weight)
                        msg += f" | grad_proj_hash={gnorm_proj:.3f}"
                    except:
                        pass
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'vm'):
                    try:
                        # VM plane gradients
                        gnorm_vm_xy = get_grad_norm(gaussians._grid.vm.xy)
                        gnorm_vm_mlp = get_grad_norm(gaussians._grid.vm.mlp.weight)
                        msg += f" | grad_vm_plane={gnorm_vm_xy:.3f} | grad_vm_mlp={gnorm_vm_mlp:.3f}"
                    except:
                        pass
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'hash'):
                    try:
                        # Hash encoder gradient (tcnn internal params)
                        gnorm_hash = get_grad_norm(gaussians._grid.hash.params)
                        msg += f" | grad_hash={gnorm_hash:.3f}"
                    except:
                        pass
                
                # Network heads gradients for comparison
                try:
                    gnorm_opacity = get_grad_norm(gaussians._opacity_head.params)
                    gnorm_frest = get_grad_norm(gaussians._features_rest_head.params)
                    msg += f" | grad_opacity={gnorm_opacity:.3f} grad_frest={gnorm_frest:.3f}"
                except:
                    pass
                
                # Feature scale value (learnable amplification factor)
                if hybrid_config and hybrid_config.hybrid_enable and hasattr(gaussians._grid, 'feature_scale'):
                    try:
                        feat_scale = float(gaussians._grid.feature_scale.item())
                        msg += f" | feat_scale={feat_scale:.2f}"
                    except:
                        pass
                
                print(msg)
            
            gaussians.optimizer.step()
            gaussians.optimizer_i.step()
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
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            else:
                if iteration % opt.prune_interval == 0:
                    gaussians.mask_prune()
            
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # [ORIG] Optimizer step (now handled by AMP scaler above)
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer_i.step()
            #     gaussians.optimizer.zero_grad(set_to_none = True)
            #     gaussians.optimizer_i.zero_grad(set_to_none = True)
            
            # [HYBRID] Optimizer step now inside AMP block above
            # No separate step needed here

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    hep = HybridEncoderParams(parser)  # Hybrid encoder config
    wlp = WaveletLossParams(parser)    # Wavelet loss config
    tap = TrainingAccelParams(parser)  # Training acceleration config
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
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from,
        hybrid_config=hep.extract(args),  # Pass hybrid encoder params
        wavelet_config=wlp.extract(args),  # Pass wavelet loss params
        accel_config=tap.extract(args)     # Pass training accel params
    )

    # All done
    print("\nTraining complete.")
