# fps_test.py
import torch
from scene import Scene
import os
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render_eval as render
import time
import numpy as np


def _validate_encoder_artifacts(model_path: str, encoder_variant: str, iteration: int | None = None):
    compression_dir = os.path.join(model_path, "compression")
    vm_planes = os.path.join(compression_dir, "vm_planes_fp16.npz")
    if not os.path.isdir(compression_dir):
        return
    config_paths = []
    if iteration is not None and iteration >= 0:
        config_paths.append(os.path.join(compression_dir, f"iteration_{iteration}", "config.npz"))
    else:
        iter_dirs = [d for d in os.listdir(compression_dir) if d.startswith("iteration_")]
        if iter_dirs:
            try:
                latest_iter = max(int(d.split("iteration_")[1]) for d in iter_dirs if d.split("iteration_")[-1].isdigit())
                config_paths.append(os.path.join(compression_dir, f"iteration_{latest_iter}", "config.npz"))
            except ValueError:
                pass

    for cfg_path in config_paths:
        if os.path.isfile(cfg_path):
            cfg = np.load(cfg_path)
            saved_variant = cfg['encoder_variant'] if 'encoder_variant' in cfg else 'hybrid'
            if saved_variant != encoder_variant:
                raise RuntimeError(
                    f"encoder_variant mismatch: artifacts={saved_variant}, requested={encoder_variant}, cfg_path={cfg_path}"
                )
            break

    has_vm = os.path.isfile(vm_planes)
    if encoder_variant == "hash_only" and has_vm:
        raise RuntimeError(f"encoder_variant=hash_only but VM planes found at {vm_planes}; pick hybrid or clean artifacts")
    if encoder_variant == "hybrid" and not has_vm:
        raise RuntimeError(f"encoder_variant=hybrid but VM planes missing at {vm_planes}; ensure run matches variant")

def measure_fps(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, optimize: bool = False, prune_opacity_th: float = 0.01, force_sh: int = -1):
    with torch.no_grad():
        # --- ECCV Ablation Config (A→B→C→D) ---
        feature_mod_type = getattr(dataset, 'feature_mod_type', 'film')
        densification_mode = getattr(dataset, 'densification_mode', 'mass_aware')
        
        encoder_variant = getattr(dataset, 'encoder_variant', 'hybrid')
        densify_strategy = 'feature_weighted' if densification_mode == 'mass_aware' else 'original_3dgs'

        _validate_encoder_artifacts(dataset.model_path, encoder_variant, iteration)
        
        print(f"[Config] Encoder: {encoder_variant}, Densify: {densify_strategy}, ModType: {feature_mod_type}")

        gaussians = GaussianModel(
            dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, 
            dataset.feature_role_split, dataset.geo_resolution, dataset.geo_rank, dataset.geo_channels,
            encoder_variant=encoder_variant,
            densify_strategy=densify_strategy,
            feature_mod_type=feature_mod_type,
        )
        # ---------------------------

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # GlowGS 优化：预计算属性以加速渲染
        print("Precomputing attributes...")
        gaussians.precompute_attributes()

        if optimize:
            print(f"\n[Optimization Enabled]")
            original_count = gaussians.get_xyz.shape[0]
            
            # 1. Opacity Pruning
            # get_opacity is sigmoid(opacity_logits), so it ranges 0-1
            opacities = gaussians.get_opacity
            # Mask should be True for points to PRUNE (opacity < threshold)
            mask = (opacities < prune_opacity_th).squeeze()
            
            if mask.sum() > 0:
                print(f"-> Pruning low opacity Gaussians (<{prune_opacity_th})")
                gaussians.prune_points(mask)
                new_count = gaussians.get_xyz.shape[0]
                reduction = 100 * (original_count - new_count) / original_count
                print(f"   Reduced from {original_count:,} to {new_count:,} (-{reduction:.1f}%)")
            else:
                print(f"-> No points pruned with opacity threshold {prune_opacity_th}")
            
            # 2. Force SH Degree (如果指定)
            # 在 render() 调用之前，我们可以修改 gaussians.active_sh_degree
            if force_sh >= 0 and force_sh < gaussians.active_sh_degree:
                print(f"-> Reducing SH Degree from {gaussians.active_sh_degree} to {force_sh}")
                gaussians.active_sh_degree = force_sh
                
            print("")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 获取测试相机
        if not skip_test:
            cameras = scene.getTestCameras()
        elif not skip_train:
            cameras = scene.getTrainCameras()
        else:
            print("No cameras selected for testing.")
            return

        if len(cameras) == 0:
            print("No cameras found!")
            return

        print(f"Starting FPS test on {len(cameras)} views...")

        # -------------------------------------------
        # 1. Warm up (预热)
        # 让 GPU 进入高性能状态，并完成 PyTorch/CUDA kernel 的初次编译
        # -------------------------------------------
        warm_up_frames = 10
        print(f"Warming up for {warm_up_frames} frames...")
        for idx in range(warm_up_frames):
            # 取第一张图反复渲染
            _ = render(cameras[0], gaussians, pipeline, background)
        torch.cuda.synchronize()

        # -------------------------------------------
        # 2. 正式计时
        # -------------------------------------------
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times = []
        
        print("Benchmarking...")
        for idx, view in enumerate(tqdm(cameras, desc="Rendering")):
            start_event.record()
            
            # --- 核心渲染调用 ---
            # 这里的 render 必须是不包含 save_image 的函数
            # 注意：如果 gaussian_renderer.render 本身包含 IO，这里需要修改
            # 但通常 Gaussian Splatting 的 render() 函数只返回 Tensor 字典
            _ = render(view, gaussians, pipeline, background)
            # --------------------

            end_event.record()
            torch.cuda.synchronize()
            
            # 记录毫秒数
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_time_ms)

        # -------------------------------------------
        # 3. 结果计算
        # -------------------------------------------
        avg_time_ms = sum(times) / len(times)
        avg_fps = 1000.0 / avg_time_ms
        
        print("\n" + "="*30)
        print(f"Results for {dataset.model_path}")
        print(f"Total Frames: {len(times)}")
        print(f"Average Time: {avg_time_ms:.2f} ms")
        print(f"Average FPS : {avg_fps:.2f} FPS")
        print("="*30 + "\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="FPS Testing Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--optimize", action="store_true", help="Apply inference-time optimizations (pruning) to boost FPS")
    parser.add_argument("--prune_opacity", type=float, default=0.01, help="Opacity threshold for pruning (default: 0.01)")
    parser.add_argument("--force_sh", type=int, default=-1, help="Force specific SH degree (0-3). -1 means use model default.")
    args = get_combined_args(parser)
    encoder_variant = getattr(args, "encoder_variant", "hybrid")
    encoder_variant_source = getattr(args, "encoder_variant_source", "default")
    # TODO(stage1-task4): keep train/render/convert/fps consistent on encoder_variant for fair ablation
    print(f"[INFO] encoder_variant={encoder_variant} (source={encoder_variant_source}) | model_dir={args.model_path} | iteration={args.iteration}")

    dataset = model.extract(args)
    dataset.encoder_variant_source = encoder_variant_source
    
    measure_fps(dataset, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.optimize, args.prune_opacity, args.force_sh)
