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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        # Ablation switches that need explicit True/False values
        ablation_switches = {'use_hybrid_encoder', 'use_edge_loss', 'use_feature_densify'}
        
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if key in ablation_switches:
                        # Ablation switches: require explicit True/False value
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, 
                                         type=lambda x: (str(x).lower() == 'true'))
                    else:
                        # Regular bool flags: presence = True
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if key in ablation_switches:
                        # Ablation switches: require explicit True/False value
                        group.add_argument("--" + key, default=value, 
                                         type=lambda x: (str(x).lower() == 'true'))
                    else:
                        # Regular bool flags: presence = True
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.pcd_path = "none"
        self.data_device = "cuda"
        self.eval = False
        # Hash grid parameters
        self.hash_size = 19          # log2_hashmap_size: 19 → 512K entries (balance of size/quality)
        self.width = 64              # MLP hidden width
        self.depth = 2               # MLP depth
        # GeoEncoder (tri-plane) parameters - tuned for size/quality balance
        self.geo_resolution = 48     # Tri-plane resolution (48x48, reduced from 64 for smaller model)
        self.geo_rank = 6            # Low-rank factorization rank (reduced from 8)
        self.geo_channels = 8        # Output feature channels
        self.feature_role_split = True  # Enable geometry/appearance feature disentanglement
        
        # Ablation switches - moved to ModelParams because they affect model structure
        self.use_hybrid_encoder = True     # Enable hybrid hash+VM encoder with role split
        self.use_edge_loss = True          # Enable unified edge-aware gradient loss
        self.use_feature_densify = True    # Enable feature-weighted densification
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.debug_equiv = False  # Equivalence debug mode: detailed config/optimizer/iteration logs
        self.debug_full_gap = False  # Full ablation gap debug mode (read-only logs)
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        # position lr
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # grid lr
        self.grid_lr_init = 0.01
        self.grid_lr_final = 0.001
        self.grid_lr_delay_steps = 5_000
        self.grid_lr_delay_mult = 0.01
        self.grid_lr_max_steps = 30_000
        # opacity lr
        self.opacity_lr_init = 0.01
        self.opacity_lr_final = 0.001
        self.opacity_lr_delay_steps = 5_000
        self.opacity_lr_delay_mult = 0.01
        self.opacity_lr_max_steps = 30_000
        # feature_rest lr
        self.feature_rest_lr_init = 0.0025
        self.feature_rest_lr_final = 0.00025
        self.feature_rest_lr_delay_steps = 5_000
        self.feature_rest_lr_delay_mult = 0.01
        self.feature_rest_lr_max_steps = 30_000
        # scaling lr
        self.scaling_lr_init = 0.01
        self.scaling_lr_final = 0.001
        self.scaling_lr_delay_steps = 5_000
        self.scaling_lr_delay_mult = 0.01
        self.scaling_lr_max_steps = 30_000
        # rotation lr
        self.rotation_lr_init = 0.002
        self.rotation_lr_final = 0.0002
        self.rotation_lr_delay_steps = 5_000
        self.rotation_lr_delay_mult = 0.01
        self.rotation_lr_max_steps = 30_000
        # features_dc lr
        self.feature_dc_lr = 0.0025
        # features_base lr
        self.scaling_base_lr = 0.005
        # mask lr
        self.mask_lr = 0.01
        self.sh_mask_lr = 0.01
        # others
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.004
        self.lambda_sh_mask = 0.0001
        
        # Densification schedule (LocoGS defaults)
        self.densification_interval = 100   # densify every N iters
        self.opacity_reset_interval = 3000  # reset opacity to prevent over-saturation
        self.densify_from_iter = 500        # start densification after warmup
        self.densify_until_iter = 15_000    # LocoGS: 15K (stop densification)
        
        # densify_grad_threshold: gradient magnitude to trigger clone/split
        # LocoGS original: 0.0002
        self.densify_grad_threshold = 0.0002
        
        # Pruning thresholds (LocoGS defaults)
        # min_opacity: prune Gaussians with opacity below this (LocoGS: 0.005 in train.py)
        self.min_opacity = 0.005
        # mask_prune_threshold: prune based on learned mask (LocoGS: 0.01 hardcoded)
        self.mask_prune_threshold = 0.01
        # prune_interval: mask-based pruning frequency after densify_until_iter (LocoGS: 1000)
        self.prune_interval = 1000
        
        self.random_background = False

        # Edge-aware loss configuration (GlowGS innovation #2)
        self.enable_edge_loss = True        # master switch for unified edge-aware gradient loss
        self.edge_loss_start_iter = 5000    # delay edge loss until geometry stabilizes (warmup)
        self.lambda_grad = 0.1              # edge loss weight (0.02~0.1 typical; 0.05 balanced)
        self.edge_flat_weight = 0.5         # flat region penalty weight (alpha in paper; 0.5 = equal)
        # Profiling 相关参数（默认关闭）
        self.profile_iters = 0
        self.profile_wait = 2
        self.profile_warmup = 2
        self.profile_active = 4
        self.profile_logdir = ""
        # Mixed precision training (AMP) - 可选启用以加速训练和减少显存
        self.use_amp = False  # Set to True to enable automatic mixed precision
        
        # Detail-aware densification/pruning (enabled by default)
        # Reallocates Gaussian budget toward high-frequency detail regions
        self.enable_detail_aware = True          # Master switch for detail-aware densify/prune
        self.detail_ema_decay = 0.9              # EMA decay for per-Gaussian detail_importance
        self.detail_importance_power_edge = 1.2  # Exponent for edge_strength in pixel_importance
        self.detail_importance_power_error = 1.0 # Exponent for error_strength in pixel_importance
        self.detail_densify_scale = 0.5          # k: effective_threshold = base / (1 + k * detail_importance)
        self.detail_prune_weight = 0.2           # Weight for detail term in prune score
        
        # Capacity control: hard cap on total Gaussian count
        # Prevents densification from exploding memory on large scenes
        self.max_gaussians = 4_000_000           # N_max: maximum number of Gaussians (4M for 48GB GPU)
        
        # ========================================================================
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    # Parse defaults separately so we can detect which CLI flags were
    # explicitly provided (vs. defaults). This prevents render/eval from
    # overwriting the training config in cfg_args with parser defaults.
    default_args = parser.parse_args([])
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        # Only override values that were explicitly set on the command line.
        # This keeps cfg_args (saved during training) authoritative for
        # ablation switches and architecture hyperparameters.
        if v != getattr(default_args, k):
            merged_dict[k] = v
    
    # Ensure all command line arguments are present in merged dict
    # (for backward compatibility with old cfg_args that may be missing new parameters)
    for k, v in vars(default_args).items():
        if k not in merged_dict:
            merged_dict[k] = v
    
    # Ensure GeoEncoder parameters have defaults for backward compatibility
    # with models trained before these parameters were added
    geo_defaults = {
        'geo_resolution': 48,
        'geo_rank': 6,
        'geo_channels': 8,
        'feature_role_split': True
    }
    for key, default_val in geo_defaults.items():
        if key not in merged_dict:
            merged_dict[key] = default_val
            print(f"[INFO] Using default {key}={default_val}")
    
    return Namespace(**merged_dict)

