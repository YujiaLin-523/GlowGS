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
from utils.general_utils import is_verbose

# ── Three ablation switches (the ONLY user-facing switches) ─────────────
# Registered with type=_bool_flag so that:
#   --enable_vm True/False
#   --enable_mass_aware True/False
#   --enable_edge_loss True/False
# Defaults are True (full GlowGS).
def _bool_flag(v):
    """Parse boolean CLI flags: True/False/1/0/yes/no."""
    return str(v).lower() in ('true', '1', 'yes')

ABLATION_SWITCHES = ('enable_vm', 'enable_mass_aware', 'enable_edge_loss')


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if key in ABLATION_SWITCHES:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value,
                                         type=_bool_flag)
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if key in ABLATION_SWITCHES:
                        group.add_argument("--" + key, default=value,
                                         type=_bool_flag)
                    else:
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
        # GeoEncoder (VM decomposition) parameters
        self.geo_resolution = 128    # VM initial resolution
        self.geo_rank = 48           # VM rank (increased for sum aggregation)
        self.geo_channels = 32       # VM output feature channels
        
        # ── Three ablation switches (unified, --flag True/False) ──
        self.enable_vm = True              # VM tri-plane branch in encoder
        self.enable_mass_aware = True      # Mass-aware densification gating
        self.enable_edge_loss = True       # Edge-aware gradient loss
        
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
        self.percent_dense = 0.005
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.001
        self.lambda_sh_mask = 0.0001
        
        # Densification schedule (LocoGS defaults)
        self.densification_interval = 100   # densify every N iters
        self.opacity_reset_interval = 3000  # reset opacity to prevent over-saturation
        self.densify_from_iter = 500        # start densification after warmup
        self.densify_until_iter = 15_000    # LocoGS: 15K (stop densification)
        # Mass-aware densification strength (xi). Larger = stronger pruning.
        self.mass_aware_scale = 0.1
        
        # Mass-Aware Gate (GlowGS innovation: block large & transparent from clone/split)
        # This prevents "garbage duplication" - copying volumetric blobs that occlude fine details
        # Gate only affects clone/split selection, NOT pruning (safe: cannot cause point collapse)
        self.enable_mass_gate = True              # Master switch for Mass Gate in clone/split
        self.mass_gate_opacity_threshold = 0.3    # Opacity below this = transparent (candidate for blocking)
        self.mass_gate_radius_percentile = 80.0   # Radius above this percentile = large (candidate for blocking)
        
        # densify_grad_threshold: gradient magnitude to trigger clone/split
        self.densify_grad_threshold = 0.0004
        
        # Pruning thresholds (LocoGS defaults)
        # min_opacity: prune Gaussians with opacity below this (LocoGS: 0.005 in train.py)
        self.min_opacity = 0.005
        # mask_prune_threshold: prune based on learned mask (LocoGS: 0.01 hardcoded)
        self.mask_prune_threshold = 0.005
        # prune_interval: mask-based pruning frequency after densify_until_iter (LocoGS: 1000)
        self.prune_interval = 1000
        
        self.random_background = False

        # Edge-aware loss configuration (GlowGS innovation #2)
        # Controlled by --enable_edge_loss True/False in ModelParams; these are schedule params.
        self.edge_loss_start_iter = 5000    # edge loss ramp start (begin transition)
        self.edge_loss_end_iter = 7000      # edge loss ramp end (full strength)
        self.lambda_grad = 0.05             # edge loss weight (cosine term is small; keep strong enough)
        self.edge_flat_weight = 0.5         # flat region penalty weight (restores background denoising)
        
        # Warmup/Ramp configuration for step-free training (avoid 5k iteration discontinuity)
        self.mass_aware_start_iter = 3000   # mass-aware gradient weighting ramp start
        self.mass_aware_end_iter = 5000     # mass-aware gradient weighting ramp end
        self.size_prune_start_iter = 4000   # size-aware pruning ramp start
        self.size_prune_end_iter = 6000     # size-aware pruning ramp end
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
    verbose = is_verbose()
    # Parse defaults separately so we can detect which CLI flags were
    # explicitly provided (vs. defaults). This prevents render/eval from
    # overwriting the training config in cfg_args with parser defaults.
    default_args = parser.parse_args([])
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        if verbose:
            print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            if verbose:
                print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        # Only override values that were explicitly set on the command line.
        # This keeps cfg_args (saved during training) authoritative for
        # ablation switches and architecture hyperparameters.
        # TODO(stage1-task2): add explicit test for CLI override precedence over cfg
        if v != getattr(default_args, k):
            merged_dict[k] = v
    
    # Ensure all command line arguments are present in merged dict
    # (for backward compatibility with old cfg_args that may be missing new parameters)
    for k, v in vars(default_args).items():
        if k not in merged_dict:
            merged_dict[k] = v
    
    # Ensure GeoEncoder parameters have defaults
    geo_defaults = {
        'geo_resolution': 128,
        'geo_rank': 48,
        'geo_channels': 32,
    }
    for key, default_val in geo_defaults.items():
        if key not in merged_dict:
            merged_dict[key] = default_val
            print(f"[INFO] Using default {key}={default_val}")

    # ── Migration: old cfg_args → new 3-switch system ──────────────────
    # Old configs may contain encoder_variant / feature_mod_type /
    # densification_mode / use_edge_loss.  Map them once so render/eval
    # always use the training-time intention.
    # When BOTH old and new keys are present (transitional cfg_args),
    # old keys take priority because they reflect actual training config.
    # Also handle the intermediate-era keys: mass_aware, edge_loss (before this rename)
    _has_old = any(k in merged_dict for k in ('encoder_variant', 'densification_mode', 'use_edge_loss', 'mass_aware', 'edge_loss'))
    _migrated = False
    if _has_old or 'enable_vm' not in merged_dict:
        if 'encoder_variant' in merged_dict:
            old_ev = merged_dict.pop('encoder_variant', 'hybrid')
            merged_dict['enable_vm'] = (old_ev != 'hash_only')
            _migrated = True
        elif 'enable_vm' not in merged_dict:
            merged_dict['enable_vm'] = True
    if _has_old or 'enable_mass_aware' not in merged_dict:
        if 'densification_mode' in merged_dict:
            old_dm = merged_dict.pop('densification_mode', 'mass_aware')
            merged_dict['enable_mass_aware'] = (old_dm == 'mass_aware')
            _migrated = True
        elif 'mass_aware' in merged_dict:
            merged_dict['enable_mass_aware'] = bool(merged_dict.pop('mass_aware'))
            _migrated = True
        elif 'enable_mass_aware' not in merged_dict:
            merged_dict['enable_mass_aware'] = True
    if _has_old or 'enable_edge_loss' not in merged_dict:
        if 'use_edge_loss' in merged_dict:
            old_el = merged_dict.pop('use_edge_loss', True)
            merged_dict['enable_edge_loss'] = bool(old_el)
            _migrated = True
        elif 'edge_loss' in merged_dict:
            merged_dict['enable_edge_loss'] = bool(merged_dict.pop('edge_loss'))
            _migrated = True
        elif 'enable_edge_loss' not in merged_dict:
            merged_dict['enable_edge_loss'] = True
    if _migrated:
        print(f"[CFG-MIGRATE] Mapped legacy config → enable_vm={merged_dict['enable_vm']} | enable_mass_aware={merged_dict['enable_mass_aware']} | enable_edge_loss={merged_dict['enable_edge_loss']}")

    # Strip stale legacy keys that must not leak into the new Namespace
    for stale in ('encoder_variant', 'encoder_variant_source', 'feature_mod_type',
                  'densification_mode', 'use_edge_loss', 'mass_aware', 'edge_loss'):
        merged_dict.pop(stale, None)

    return Namespace(**merged_dict)
