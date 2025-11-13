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

# ========================================
# DEBUG Configuration (Centralized)
# ========================================
class DEBUG_CFG:
    """
    Centralized debug configuration for gradient/fusion diagnostics.
    All debug code is guarded by these flags to ensure zero overhead when disabled.
    
    Default OFF to avoid accidental memory pressure and potential numeric side effects
    when collecting graph tensors for debug.
    """
    ON = False         # Master switch: set False to disable all debug output
    EVERY = 1000       # Print frequency: check every N training steps
    GRAD_NORM = True   # Print gradient norms for key parameters
    FUSION = True      # Print beta/gate statistics
    NUMERIC = True     # Check for NaN/Inf in intermediate tensors
    PERF = True        # Monitor step time and memory usage

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
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
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
        self.sh_degree = 4                  # Upgraded from 3 to 4 for drjohnson specular details
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.pcd_path = "none"
        self.data_device = "cuda"
        self.eval = False
        self.hash_size = 20         # 2^20 = 1M entries [RESTORED] Keep original capacity
        self.width = 64
        self.depth = 2
        # No initial point cloud downsampling - use full SfM output for healthy densification
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
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        # position lr
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # grid lr - increased for stronger learning
        self.grid_lr_init = 0.016      # Increased from 0.01 (60% boost)
        self.grid_lr_final = 0.0016    # Increased from 0.001 (60% boost)
        self.grid_lr_delay_steps = 3_000  # Faster ramp-up (was 5000)
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
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.random_background = False
        # LocoGS original densification parameters for healthy point cloud growth
        # [FIX CRITICAL-1] Extend densification window to allow late-stage refinement
        # Training log shows point cloud collapse (300万→60万) after 15k iter
        self.densify_until_iter = 25_000  # Extended to 83% of training (was 50%)
        self.densify_grad_threshold = 0.00014  # Reduced 30% to encourage fine detail splits
        # Standard pruning parameters
        self.prune_interval = 100      # Prune every densification cycle
        self.prune_opacity_threshold = 0.005  # LocoGS original value
        
        # [UNIVERSAL ENHANCEMENT] Adaptive densification for fine details & high-contrast scenes
        # These are NOT specific to any dataset, but help all scenes with:
        # - High-frequency details (specular highlights, thin structures)
        # - Fine edges and boundaries (windows, door frames)
        # - Avoid over-smoothing while ensuring point count stability
        
        self.densify_grad_threshold_factor = 0.7   # Reduce grad threshold to 70% → catch finer splits
        self.densify_until_iter_factor = 1.5       # Extend densification to 150% of iterations (45k for 30k training)
        self.size_threshold_multiplier = 1.3       # Multiply size threshold ×1.3 → prune paint rollers sooner
        self.prune_opacity_threshold_final = 0.01  # Stricter final opacity (0.01 vs default 0.005) → clean semi-transparent tails
        
        super().__init__(parser, "Optimization Parameters")



class HybridEncoderParams(ParamGroup):
    """
    Hybrid encoder: VM (low-freq) + HashGrid (high-freq)
    All new knobs live here to keep diffs small and audits simple.
    """
    def __init__(self, parser):
        # Enable/disable
        self.hybrid_enable = True
        
        # [UNIVERSAL ENHANCEMENTS] Global master switch for all improvements
        # Set to False to test baseline performance (for ablation studies)
        self.universal_enhancements_enable = True
        
        # VM encoder - [CAPACITY BOOST] Increase to reduce "paint roller" artifacts
        # Checklist A4: larger planes reduce horizontal/vertical streaking
        self.vm_rank = 18              # +12.5% capacity (16→18) for finer low-freq details
        self.vm_plane_res = 224        # +17% resolution (192→224) to reduce grid aliasing
        self.vm_out_dim = 32           # Output dimension
        self.vm_basis = 'bilinear'     # Sampling: 'bilinear' or 'nearest'
        # [FIX C-2] Gradient checkpointing trades compute for memory with no API changes.
        # Enable (True) for large scenes to reduce activation memory at the cost of recomputation.
        self.vm_checkpoint = False     # Gradient checkpointing (disable for speed)
        
        # Fusion: convex blend with floor to ensure both branches always get gradients
        # [ORIG] self.fusion_mode = 'residual'
        # [ORIG] self.init_hash_gain = 1.0
        self.fusion_mode = 'convex'    # 'convex' blend ensures both branches get gradients
        self.init_hash_gain = 0.05     # Start very small to give VM strong early learning
        # [FIX] Extend warm-up to give VM more learning time before Hash dominance
        self.warm_steps = 10000        # Extended to 33% of training (was 23%)
        
        # Fusion floor: beta = beta_min + (1-2*beta_min)*beta_raw ensures VM always gets gradients
        # [FIX CRITICAL-3] Increase VM minimum weight to prevent gradient starvation
        # Training log: vm_xy gradient drops 20x after hash_gain=1.0 (7k iter)
        self.beta_min_start = 0.15     # Higher early floor (30% VM minimum → stronger gradients)
        self.beta_min_end = 0.03       # Lower late floor (allow Hash to dominate when appropriate)
        
        # Gate annealing (FIX: prevents HF branch from dominating too early)
        # Gate formula: m_hf = sigmoid(alpha * (log2(kappa / sigma_max) - tau))
        # [ORIG] self.gate_alpha = 8.0
        # [ORIG] self.gate_tau = 0.0
        self.gate_alpha_start = 3.0    # Initial gate sharpness (gentle)
        self.gate_alpha_end = 8.0      # Final gate sharpness (sharp)
        self.gate_tau_start = 1.0      # Initial threshold offset (biased toward VM)
        self.gate_tau_end = 0.0        # Final threshold offset (neutral)
        self.gate_kappa = 1.0          # Gate scale reference (normalized)
        self.gate_clamp = (0.0, 1.0)   # Clamp gate output to valid range
        
        # Feature scaling (FIX: tame instability)
        # [ORIG] feature_scale was trainable and could grow unbounded
        self.feature_scale_trainable = False  # Disable trainable scaling (use constant 1.0)
        self.feature_scale_init = 1.0         # Safe default (no amplification)
        self.feature_scale_clamp = (0.1, 10.0)  # Clamp range if trainable
        
        # VM plane forward-only low-pass filter (stabilizes training without altering parameters)
        self.vm_lpf_enable = True      # Enable forward-only low-pass on VM planes
        self.vm_lpf_kernel = 3         # Kernel size (3x3 avg pool)
        self.vm_lpf_weaken_steps = 5000  # Steps to gradually reduce LPF strength (optional)
        
        # Memory optimization (batching only when constrained)
        self.update_batch_size = 100000  # Chunk size for micro-batching when memory is tight
        self.oom_threshold = 3000000     # Only batch if N > this threshold (3M points)
                                         # Below threshold: full-batch training for better performance
                                         # Above threshold: chunked forward to avoid OOM
        
        # VM optimizer settings (FIX: use param groups instead of manual grad scaling)
        # [ORIG] Manual grad *= k hacks removed; use optimizer param groups instead
        # [FIX] CHECKLIST A7: Reduce VM LR multiplier for drjohnson stability
        self.vm_lr_multiplier = 6.0    # Reduced from 15× → 6× (balanced stability + gradients)
        self.vm_weight_decay = 0.0     # No weight decay for VM planes (preserve capacity)
        
        # [ORIG] Staged training removed (unnecessary with convex blend + warm-up)
        super().__init__(parser, "Hybrid Encoder Parameters")


class WaveletLossParams(ParamGroup):
    """Wavelet loss for high-frequency details (requires: pip install pytorch_wavelets)"""
    def __init__(self, parser):
        # [FIX CHECKLIST A8] Enable wavelet loss with delayed gradual ramp (3k->8k)
        self.wavelet_enable = True      # Enable to constrain high-frequency details
        # [NEW] Gradual ramp: prevent conflict with VM LPF + gate transition
        self.wavelet_start_iter = 3000  # Start ramping at 3k (was fixed 5k)
        self.wavelet_ramp_iter = 8000   # Finish ramp at 8k (linear from 0 to wavelet_lambda_h)
        self.wavelet_levels = 2         # Decomposition levels
        self.wavelet_lambda_h = 0.4     # High-freq weight (moderate, was 1.0)
        self.wavelet_lambda_ll = 0.0    # Disable low-freq (avoid VM LPF conflict)
        self.wavelet_grayscale = True   # Convert to grayscale
        super().__init__(parser, "Wavelet Loss Parameters")


class VMauxLossParams(ParamGroup):
    """VM low-frequency auxiliary loss configuration (VM-only backprop)"""
    def __init__(self, parser):
        self.vm_aux_enable = False     # Disabled by default for stable baseline
        self.vm_aux_lambda = 0.3       # Weight for VM auxiliary loss
        self.vm_aux_blur_k_init = 7    # Initial blur kernel size for GT low-pass
        self.vm_aux_blur_k_end = 3     # Final blur kernel size (curriculum)
        self.vm_aux_blur_steps = 5000  # Steps to anneal blur kernel size
        super().__init__(parser, "VM Auxiliary Loss Parameters")


class OrthogonalityParams(ParamGroup):
    """Feature decorrelation/orthogonality constraint between VM and Hash branches"""
    def __init__(self, parser):
        self.orth_enable = False       # Disabled by default for stable baseline
        self.orth_lambda = 1e-3        # Weight for orthogonality penalty
        super().__init__(parser, "Orthogonality Loss Parameters")


class KappaAdaptiveParams(ParamGroup):
    """
    Delayed kappa initialization for reliable scale-based frequency gating.
    
    Problem: Fixed kappa=1.0 at iteration 0 is unreliable when initial point cloud
    has no scale statistics. This causes frequency gate to misdirect VM/Hash balance.
    
    Solution: Delay kappa calibration until iteration ~1000, then compute from 
    log2(sigma_max) distribution (median + tunable offset delta).
    
    Universally useful: Prevents early gradient instabilities in ALL scenes, especially
    those with large scale variance (wide depth range, mixed detail levels).
    """
    def __init__(self, parser):
        self.kappa_adaptive_enable = True           # Enable by default (benefits all scenes)
        self.kappa_init_iter = 1000                 # Collect stats for first 1k iterations
        self.kappa_delta = 0.0                      # Offset from median in log2 space: -0.5, 0.0, +0.5
        self.kappa_warmup_iter = 500                # Cold-start phase with weak VM/Hash
        self.kappa_beta_min_warmup = 0.15           # VM floor during warmup (prevent early gradient starvation)
        super().__init__(parser, "Kappa Adaptive Parameters")


class AnisotropyConstraintParams(ParamGroup):
    """
    Anisotropy constraint: prevent elongated paint-roller artifacts.
    
    Problem: Without constraint, Gaussians become extremely elongated, especially
    at boundaries, causing horizontal/vertical streaking at walls, doors, thin structures.
    
    Solution: Hard clamp aspect ratio + soft regularizer to prevent over-smoothing.
    Universally useful: Reduces artifacts in ALL scenes without sacrificing quality,
    especially for bounded/interior scenes with clear geometry (walls, frames, corners).
    """
    def __init__(self, parser):
        self.aniso_constraint_enable = True         # Enable by default (universal benefit)
        self.aniso_r_max = 20.0                     # Hard aspect ratio limit
        self.aniso_r_threshold = 12.0               # Soft regularizer threshold
        self.aniso_lambda = 0.01                    # Regularizer weight
        self.aniso_lambda_anneal_start = 5000       # Start annealing at 5k
        self.aniso_lambda_anneal_end = 15000        # Finish annealing at 15k (decay to 0)
        super().__init__(parser, "Anisotropy Constraint Parameters")


class NumericStabilityParams(ParamGroup):
    """Numeric stability safeguards (FP32-only, no AMP)"""
    def __init__(self, parser):
        self.log2_eps = 1e-8           # Epsilon for log2 operations
        self.sigma_min = 1e-6          # Minimum sigma_max clamp value
        self.grad_clip_norm = 1.0      # Gradient clipping norm
        self.nan_guard_enable = True   # Enable NaN fuse (skip step if NaN grad detected)
        super().__init__(parser, "Numeric Stability Parameters")


class LocalityLossParams(ParamGroup):
    """
    Locality regularizer leveraging neighboring Gaussians' attribute similarity.
    
    Encodes the LocoGS prior: spatially close Gaussians should have similar attributes
    (opacity, scale, rotation, SH residuals). Computed only on visible subset per step
    for efficiency. Improves test-view generalization on sparse-view scenes.
    
    Conservative baseline tuning for stable hybrid training:
    - Moderate loc_lambda (0.05) for gentle spatial smoothness
    - Moderate k=8 balancing quality and speed
    - Balanced attribute weights to prevent overfitting details
    """
    def __init__(self, parser):
        self.loc_enable = True         # Enable locality loss (critical for generalization)
        # [FIX CHECKLIST A7] Add annealing schedule to avoid over-smoothing in late stage
        self.loc_lambda = 0.05         # Initial weight (will be annealed in trainer)
        self.loc_lambda_end = 0.01     # Final weight after annealing
        self.loc_anneal_start = 2000   # Start annealing at 2k iter
        self.loc_anneal_end = 15000    # Finish annealing at 15k iter
        self.loc_k = 8                 # Balance quality/speed (k=12 too slow for 100万+)
        self.loc_sigma = 0.03          # Kernel width
        # Per-attribute weights (balanced for generalization)
        self.w_opacity = 1.2           # Opacity consistency important
        self.w_scale = 1.0             # Keep scale similarity
        self.w_rot = 0.6               # More rotation alignment
        self.w_sh = 0.2                # More SH smoothness
        self.max_points = 5000         # Reduce sampling for speed (was 8000)
        super().__init__(parser, "Locality Loss Parameters")


class FeatureRegParams(ParamGroup):
    """
    Feature norm regularization to prevent encoder overfitting.
    
    Penalizes large feature norms from the encoder, encouraging the model to use
    more compact, generalizable representations. Critical for test-view quality.
    """
    def __init__(self, parser):
        self.feat_reg_enable = True    # Enable feature norm regularization
        # [FIX CRITICAL-2] Reduce reg strength and delay start (training log: 34% of loss at iter 1k!)
        self.feat_reg_lambda = 0.0003  # Reduced 70% to avoid crushing encoder capacity
        self.feat_reg_start = 5000     # Delayed start: let encoder learn basic structure first
        super().__init__(parser, "Feature Regularization Parameters")


class TrainingAccelParams(ParamGroup):
    """Training acceleration: TF32, torch.compile, micro-batching, logging (NO AMP)"""
    def __init__(self, parser):
        # Mixed precision - DISABLED for stability
        self.amp_enable = False        # CRITICAL FIX: AMP causes NaN with hybrid encoder (MUST stay False)
        self.amp_dtype = 'bfloat16'    # 'bfloat16' preferred (fallback to 'float16' if unsupported) - UNUSED
        
        # TF32 acceleration (FP32 with faster matmul on Ampere+ GPUs)
        self.tf32_enable = True        # Enable TF32 for faster GEMM without AMP
        
        # torch.compile (experimental, PyTorch 2.0+)
        # [FIX] DISABLED by default - causes OOM on first run due to compilation overhead
        self.torch_compile = False     # Set to True only if you have 40GB+ VRAM
        
        # Micro-batching (removed grad_clip_norm - now in NumericStabilityParams)
        self.micro_batch_points = 0    # 0 = disabled (micro-batching for encoder forward only)
        
        # Logging
        self.log_every = DEBUG_CFG.EVERY            # Log every N steps
        super().__init__(parser, "Training Acceleration Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
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
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
