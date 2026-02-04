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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_eval as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import time


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
        raise RuntimeError(f"encoder_variant=hash_only but found VM planes at {vm_planes}; use hybrid or clean artifacts")
    if encoder_variant == "hybrid" and not has_vm:
        raise RuntimeError(f"encoder_variant=hybrid but VM planes missing at {vm_planes}; ensure you loaded the correct run")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        # ECCV Ablation Config (A→B→C→D)
        feature_mod_type = getattr(dataset, 'feature_mod_type', 'film')
        densification_mode = getattr(dataset, 'densification_mode', 'mass_aware')

        encoder_variant = getattr(dataset, 'encoder_variant', 'hybrid')
        densify_strategy = 'feature_weighted' if densification_mode == 'mass_aware' else 'original_3dgs'

        _validate_encoder_artifacts(dataset.model_path, encoder_variant, iteration)

        gaussians = GaussianModel(
            dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth, 
            dataset.feature_role_split, dataset.geo_resolution, dataset.geo_rank, dataset.geo_channels,
            encoder_variant=encoder_variant,
            densify_strategy=densify_strategy,
            feature_mod_type=feature_mod_type,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Precompute attributes once for fast rendering (skip per-frame MLP inference)
        gaussians.precompute_attributes()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    import sys
    sys.stderr.write(f"Loading scene for rendering: {args.model_path}...\n")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    encoder_variant = getattr(args, "encoder_variant", "hybrid")
    encoder_variant_source = getattr(args, "encoder_variant_source", "default")
    # TODO(stage1-task4): keep train/render/convert/fps consistent on encoder_variant for fair ablation
    print(f"[INFO] encoder_variant={encoder_variant} (source={encoder_variant_source}) | model_dir={args.model_path} | iteration={args.iteration}")

    dataset = model.extract(args)
    dataset.encoder_variant_source = encoder_variant_source

    render_sets(dataset, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)