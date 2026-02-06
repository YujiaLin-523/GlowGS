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
import numpy as np
import json
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, contract_to_unisphere, mortonEncode
from utils.quantization_utils import half_ste, quantize, dequantize
from utils.gpcc_utils import encode_xyz, decode_xyz
from encoders import create_gaussian_encoder, get_encoder_output_dims
import tinycudann as tcnn
from utils.general_utils import is_verbose

def _to_np_fp16(t: torch.Tensor):
    """Detach tensor to CPU and store as float16 for lossless downstream reload."""
    return t.detach().cpu().half().numpy()


def _save_npz(path: str, **arrays):
    """Wrapper to write compressed npz with clear intent."""
    np.savez_compressed(path, **arrays)


def _load_npz(path: str):
    return np.load(path)


def _np_to_torch(arr, device, dtype):
    return torch.from_numpy(np.asarray(arr)).to(device=device, dtype=dtype)


def _strict_assign(dst: torch.Tensor, src: torch.Tensor, name: str, path: str):
    if dst.shape != src.shape:
        raise RuntimeError(f"Shape mismatch for {name}: dst {dst.shape} vs src {src.shape} from {path}")
    if dst.numel() != src.numel():
        raise RuntimeError(f"Numel mismatch for {name}: dst {dst.numel()} vs src {src.numel()} from {path}")
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _strict_assign_param(head, src: torch.Tensor, name: str, path: str):
    dst = head.params
    if dst.shape != src.shape:
        raise RuntimeError(f"Shape mismatch for {name}: dst {dst.shape} vs src {src.shape} from {path}")
    if dst.numel() != src.numel():
        raise RuntimeError(f"Numel mismatch for {name}: dst {dst.numel()} vs src {src.numel()} from {path}")
    head.params = nn.Parameter(src.to(device=dst.device, dtype=dst.dtype))
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_base_activation = torch.exp
        self.scaling_base_inverse_activation = torch.log

        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.xyz_activation = half_ste

        self.mask_activation = torch.sigmoid
        self.sh_mask_activation = torch.sigmoid

    def setup_configs(self, hash_size, width, depth):
        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": hash_size,
            "base_resolution": 16,
            "per_level_scale": 1.447269237440378,
        }
        self.network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": width,
            "n_hidden_layers": depth,
        }
    
    def setup_params(self):
        # -----------------------------------------------------------------
        # Capacity control: hard cap on total Gaussian count to prevent
        # densification from exploding memory. Densify stops once N >= max.
        # -----------------------------------------------------------------
        self.gaussian_capacity_config = {
            "max_point_count": 6_000_000,      # Hard cap: 6M Gaussians
            "densify_until_iter": 15000,       # Match baseline schedule
            "prune_interval": 400,             # Periodic pruning interval
        }
        
        # Create encoder using factory (always hybrid, VM bypassed if enable_vm=False)
        self._grid = create_gaussian_encoder(
            encoding_config=self.encoding_config,
            network_config=self.network_config,
            geo_resolution=self._geo_resolution,
            geo_rank=self._geo_rank,
            geo_channels=self._geo_channels,
            enable_vm=self._enable_vm,
        )
        # Move encoder and its submodules to GPU
        self._grid = self._grid.cuda()
        
        # MLP heads for Gaussian attributes
        self._features_rest_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=(self.max_sh_degree + 1) ** 2 * 3 - 3, network_config=self.network_config)
        self._scaling_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=3, network_config=self.network_config)
        self._rotation_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=4, network_config=self.network_config)
        self._opacity_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=1, network_config=self.network_config)
        
        self._aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float, device='cuda')
        # Lightweight projection layers for feature_role_split path
        # Map geometry_latent/appearance_latent (C_shared + C_role) to head input dim (C_shared)
        # Only used when encoder variant supports role split (hybrid with feature_role_split=True)
        base_dim, geometry_dim, appearance_dim = get_encoder_output_dims(self._grid)
        print(f"[EncoderDims] enable_vm={self._enable_vm} role_split={self._feature_role_split} base_dim={base_dim} geometry_dim={geometry_dim} appearance_dim={appearance_dim}")
        # TODO(stage1-task3): if feature_role_split enabled, print (geometry_dim, appearance_dim, fused_dim) and assert downstream expectations
        if self._feature_role_split:
            if geometry_dim != base_dim or appearance_dim != base_dim:
                raise ValueError(
                    f"Role-split dims mismatch: base_dim={base_dim}, geometry_dim={geometry_dim}, appearance_dim={appearance_dim}"
                )
        
        if geometry_dim != base_dim:  # Role split is active
            # Projection for geometry heads (scale, rotation, opacity)
            self._geometry_input_proj = nn.Linear(geometry_dim, base_dim, bias=True).cuda()
            # Projection for appearance head (SH)
            self._appearance_input_proj = nn.Linear(appearance_dim, base_dim, bias=True).cuda()
            nn.init.normal_(self._geometry_input_proj.weight, std=0.01)
            nn.init.zeros_(self._geometry_input_proj.bias)
            nn.init.normal_(self._appearance_input_proj.weight, std=0.01)
            nn.init.zeros_(self._appearance_input_proj.bias)
            # Mark as role-split mode for update_attributes
            self._use_role_split = True
        else:
            # Fused latent mode: no separate projections needed
            self._geometry_input_proj = None
            self._appearance_input_proj = None
            self._use_role_split = False
        self._rotation_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device='cuda')
        self._opacity_init = torch.tensor([[np.log(0.1 / (1 - 0.1))]], dtype=torch.float, device='cuda')

    def __init__(self, sh_degree: int, hash_size=19, width=64, depth=2, feature_role_split=True,
                 geo_resolution=128, geo_rank=48, geo_channels=32,
                 enable_vm: bool = True,
                 enable_mass_aware: bool = True,
                 mass_aware_scale: float = 0.1,
                 enable_mass_gate: bool = True,
                 mass_gate_opacity_threshold: float = 0.3,
                 mass_gate_radius_percentile: float = 80.0,
                 # Legacy kwargs accepted but ignored (backward compat)
                 encoder_variant: str | None = None,
                 densify_strategy: str | None = None,
                 feature_mod_type: str | None = None,
                 mass_aware: bool | None = None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # ── Ablation switches ──
        self._enable_vm = enable_vm
        # Accept legacy 'mass_aware' kwarg for backward compat
        self._enable_mass_aware = enable_mass_aware if mass_aware is None else mass_aware
        self._feature_role_split = feature_role_split
        # Mass-aware gradient weighting scale (xi): higher = stronger pruning, lower = softer
        self._mass_aware_scale = mass_aware_scale
        # Mass Gate parameters (GlowGS innovation: block large & transparent from clone/split)
        self._enable_mass_gate = enable_mass_gate if self._enable_mass_aware else False
        self._mass_gate_opacity_threshold = mass_gate_opacity_threshold
        self._mass_gate_radius_percentile = mass_gate_radius_percentile
        # Alert threshold for large prune events (fraction of points removed in one call)
        self._prune_alert_thresh = 0.20
        # Store GeoEncoder parameters for serialization/deserialization
        self._geo_resolution = geo_resolution
        self._geo_rank = geo_rank
        self._geo_channels = geo_channels
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling_base = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self._sh_mask = torch.empty(0)
        self._grid = torch.empty(0)
        self._features_rest_head = torch.empty(0)
        self._scaling_head = torch.empty(0)
        self._rotation_head = torch.empty(0)
        self._opacity_head = torch.empty(0)
        self._aabb = torch.empty(0)
        self._rotation_init = torch.empty(0)
        self._opacity_init = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.optimizer_i = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Debug scale flag (env controlled, default off)
        self._debug_scale = os.getenv("DEBUG_SCALE", "0").lower() in ("1", "true", "yes")
        # Mass-aware debug flag (env controlled, default off)
        self._debug_mass = os.getenv("DEBUG_MASS", "0").lower() in ("1", "true", "yes")
        # Optional birth iteration tracker (only allocated when DEBUG_MASS=1)
        self._birth_iter = None
        # Pending densify slices (start, end, iter) for deferred stat logging
        self._pending_densify_slices = []
        # Precomputed attribute cache for fast inference (populated by precompute_attributes)
        self._precomputed_cache = None
        self.setup_functions()
        self.setup_configs(hash_size, width, depth)
        self.setup_params()
        # Compact densify/prune telemetry (reset each summary)
        self._init_dp_stats()


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._mask,
            self._sh_mask,
            self._grid,
            self._features_rest_head,
            self._scaling_head,
            self._rotation_head,
            self._opacity_head,
            self._aabb,
            self._rotation_init,
            self._opacity_init,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.optimizer_i.state_dict(),
            self.spatial_lr_scale,
        )
    
    def capture_for_eval(self):
        """
        Lightweight capture for evaluation/inference only.
        Excludes training-specific data (optimizer states, gradients) to reduce model size.
        Saves ~3-5MB compared to full capture().
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._mask,
            self._sh_mask,
            self._grid,
            self._features_rest_head,
            self._scaling_head,
            self._rotation_head,
            self._opacity_head,
            self._aabb,
            self._rotation_init,
            self._opacity_init,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        """
        Restore model from checkpoint. 
        Automatically detects lightweight (eval-only) vs full (training) checkpoints.
        """
        # Check if this is a lightweight checkpoint (16 items) or full checkpoint (21 items)
        if len(model_args) == 16:
            # Lightweight checkpoint from capture_for_eval()
            print("[INFO] Loading lightweight checkpoint (evaluation-only)")
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._mask,
            self._sh_mask,
            self._grid,
            self._features_rest_head,
            self._scaling_head,
            self._rotation_head,
            self._opacity_head,
            self._aabb,
            self._rotation_init,
            self._opacity_init,
            self.spatial_lr_scale) = model_args
            
            # Initialize training-specific buffers
            N = self._xyz.shape[0]
            self.max_radii2D = torch.zeros((N,), device="cuda")
            xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
            denom = torch.zeros((N, 1), device="cuda")
            
            # Setup training (creates optimizers)
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            # Optimizer states will be fresh (not loaded)
            print(f"[INFO] Initialized {N} Gaussians with fresh optimizer states")
            
        else:
            # Full checkpoint from capture()
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._mask,
            self._sh_mask,
            self._grid,
            self._features_rest_head,
            self._scaling_head,
            self._rotation_head,
            self._opacity_head,
            self._aabb,
            self._rotation_init,
            self._opacity_init,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            opt_i_dict,
            self.spatial_lr_scale) = model_args
            
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            self.optimizer_i.load_state_dict(opt_i_dict)
    
    def restore_for_eval(self, model_args):
        """
        Lightweight restore for evaluation/inference only.
        Compatible with capture_for_eval() output.
        Initializes training buffers to empty tensors (not used in eval mode).
        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc,
        self._features_rest,
        self._scaling_base,
        self._mask,
        self._sh_mask,
        self._grid,
        self._features_rest_head,
        self._scaling_head,
        self._rotation_head,
        self._opacity_head,
        self._aabb,
        self._rotation_init,
        self._opacity_init,
        self.spatial_lr_scale) = model_args
        
        # Initialize training buffers to empty (not needed for eval)
        N = self._xyz.shape[0]
        self.max_radii2D = torch.zeros((N,), device="cuda")
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        
        print(f"[INFO] Restored {N} Gaussians for evaluation (lightweight mode)")

    def update_attributes(self, force_update=False, iteration=None):
        """
        Update implicit Gaussian attributes from encoder output.
        
        Feature role split routing:
            - geometry_latent → scale, rotation, opacity heads
            - appearance_latent → SH/color head (features_rest)
        
        Args:
            force_update: If True, bypass cache and force recomputation
        """
        # Fast path: use precomputed cache if available (for inference)
        if self._precomputed_cache is not None and not force_update:
            self._opacity = self._precomputed_cache['opacity']
            self._scaling = self._precomputed_cache['scaling']
            self._rotation = self._precomputed_cache['rotation']
            if 'features_rest' in self._precomputed_cache:
                self._features_rest = self._precomputed_cache['features_rest']
            return
        
        # Smart caching: Only recompute if xyz changed or during densification
        # This saves ~28ms per iteration (profiler shows 569ms / 20 calls)
        N = self._xyz.shape[0]
        if N == 0:
            return

        # Training needs a fresh graph every iteration; disable caching when grads are enabled
        caching_enabled = not torch.is_grad_enabled()
        if not caching_enabled:
            force_update = True  # always recompute under autograd to avoid stale graphs
            self._cached_attributes_hash = None  # invalidate cache when switching back to training
            
        # Check if we can use cached attributes
        if caching_enabled and not force_update and hasattr(self, '_cached_attributes_hash'):
            # Use xyz tensor data pointer as a cheap hash (changes on densify/prune/optimizer step)
            current_hash = (self._xyz.data_ptr(), N, self._xyz.requires_grad)
            if current_hash == self._cached_attributes_hash:
                # Cache hit - skip expensive recomputation
                if not hasattr(self, '_cache_hit_count'):
                    self._cache_hit_count = 0
                self._cache_hit_count += 1
                return
        
        # Cache miss - need to recompute
        if not hasattr(self, '_cache_miss_count'):
            self._cache_miss_count = 0
        self._cache_miss_count += 1
        
        # Hybrid encoder always does its own L-inf contraction via AABB
        raw_xyz = self.get_xyz.detach()

        # Optimize chunk size: use larger chunks or process all at once if memory allows
        # For better GPU utilization, try to process in fewer, larger chunks
        chunk_size = min(131072, N) if N > 65536 else N  # Process all at once if < 65k, else use 128k chunks

        if self._use_role_split:
            # Collect outputs to avoid in-place writes that may invalidate autograd versions
            opacity_chunks = []
            scaling_chunks = []
            rotation_chunks = []
            features_rest_chunks = [] if self.max_sh_degree > 0 else None

            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                coords = raw_xyz[i: end_idx]
                encoder_out = self._grid(coords, self._aabb)

                # Support both 2-tuple legacy and 3-tuple new encoder outputs
                if isinstance(encoder_out, tuple):
                    if len(encoder_out) == 3:
                        shared_chunk, geometry_chunk, appearance_chunk = encoder_out
                    elif len(encoder_out) == 2:
                        geometry_chunk, appearance_chunk = encoder_out
                        shared_chunk = geometry_chunk
                else:
                    # Unexpected fused output; treat as fused latent
                    fused = encoder_out
                    fused = torch.clamp(fused, -10.0, 10.0)
                    try:
                        self._opacity[i:end_idx] = self._opacity_head(fused) + self._opacity_init
                        self._scaling[i:end_idx] = self._scaling_head(fused)
                        self._rotation[i:end_idx] = self._rotation_head(fused) + self._rotation_init
                        if self.max_sh_degree > 0:
                            self._features_rest[i:end_idx] = self._features_rest_head(fused).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
                    except Exception:
                        # Fallback: zero initialization
                        self._opacity[i:end_idx] = self._opacity_init.expand(end_idx - i, -1)
                        self._scaling[i:end_idx] = 0.0
                        self._rotation[i:end_idx] = self._rotation_init.expand(end_idx - i, -1)
                    continue

                # Geometry heads: scale, rotation, opacity
                # Project geometry_latent (C_shared + C_role) to head input dim (C_shared)
                geometry_proj = self._geometry_input_proj(geometry_chunk)  # [N, C_shared]
                geometry_proj = torch.clamp(geometry_proj, -10.0, 10.0)
                
                # Append outputs; avoid in-place writes to keep autograd versions clean
                scaling_chunks.append(self._scaling_head(geometry_proj))
                rotation_chunks.append(self._rotation_head(geometry_proj) + self._rotation_init)
                opacity_chunks.append(self._opacity_head(geometry_proj) + self._opacity_init)

                # Appearance / SH head
                if self.max_sh_degree > 0:
                    appearance_proj = self._appearance_input_proj(appearance_chunk)  # [N, C_shared]
                    appearance_proj = torch.clamp(appearance_proj, -10.0, 10.0)
                    features_rest_chunks.append(
                        self._features_rest_head(appearance_proj).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
                    )

            # Concatenate chunk outputs
            self._opacity = torch.cat(opacity_chunks, dim=0)
            self._scaling = torch.cat(scaling_chunks, dim=0)
            self._rotation = torch.cat(rotation_chunks, dim=0)
            if self.max_sh_degree > 0 and features_rest_chunks is not None:
                self._features_rest = torch.cat(features_rest_chunks, dim=0)
            # Update cache hash after successful computation
            self._cached_attributes_hash = (self._xyz.data_ptr(), N, self._xyz.requires_grad)
        else:
            # Fallback: single fused latent for all heads
            # Pre-allocate output tensors for better performance
            opacity_chunks = []
            scaling_chunks = []
            rotation_chunks = []
            features_rest_chunks = [] if self.max_sh_degree > 0 else None
            
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                coords = raw_xyz[i: end_idx]
                encoder_out = self._grid(coords, self._aabb)
                feats = encoder_out if not isinstance(encoder_out, tuple) else encoder_out[0]
                # Encoder already clamps, skip redundant clamp
                opacity_chunks.append(self._opacity_head(feats) + self._opacity_init)
                scaling_chunks.append(self._scaling_head(feats))
                rotation_chunks.append(self._rotation_head(feats) + self._rotation_init)
                if self.max_sh_degree > 0:
                    features_rest_chunks.append(self._features_rest_head(feats).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3))

            self._opacity = torch.cat(opacity_chunks, dim=0)
            self._scaling = torch.cat(scaling_chunks, dim=0)
            self._rotation = torch.cat(rotation_chunks, dim=0)
            if self.max_sh_degree > 0 and features_rest_chunks is not None:
                self._features_rest = torch.cat(features_rest_chunks, dim=0)
            
            # Update cache hash after successful computation
            self._cached_attributes_hash = (self._xyz.data_ptr(), N, self._xyz.requires_grad)
        
        # NOTE: Removed torch.cuda.empty_cache() here for real-time performance
        # empty_cache() is expensive (~1-2ms) and should only be called when necessary
        self._debug_scale_stats(tag="after_update_attributes", tensor=self._scaling_base, iteration=iteration, force=False)

    def precompute_attributes(self):
        """
        Precompute and cache all Gaussian attributes for fast inference.
        
        Call this once after model loading to avoid per-frame MLP inference.
        The cached attributes will be used by update_attributes() automatically.
        """
        
        # Force compute all attributes
        self._precomputed_cache = None  # Clear any existing cache
        self.update_attributes(force_update=True)
        
        # Cache the computed attributes (detach to save memory)
        self._precomputed_cache = {
            'opacity': self._opacity.detach().clone(),
            'scaling': self._scaling.detach().clone(),
            'rotation': self._rotation.detach().clone(),
        }
        if self.max_sh_degree > 0 and self._features_rest.numel() > 0:
            self._precomputed_cache['features_rest'] = self._features_rest.detach().clone()
        
        n_points = self._xyz.shape[0]
        if is_verbose():
            print(f"[Precompute] Cached {n_points:,} Gaussian attributes for fast rendering")
        self._debug_scale_stats(tag="after_precompute", tensor=self._scaling_base, iteration=None, force=True)
    
    def _debug_scale_enabled(self):
        # Env flag takes precedence; attribute can be toggled by caller
        if os.getenv("DEBUG_SCALE", "0").lower() in ("1", "true", "yes"):
            return True
        return bool(getattr(self, "_debug_scale", False))

    def _sample_scale_stats(self, tensor: torch.Tensor, sample: int = 8192):
        if tensor is None or tensor.numel() == 0:
            return None
        N = tensor.shape[0]
        s = min(sample, N)
        if s <= 0:
            return None
        idx = torch.randint(0, N, (s,), device=tensor.device)
        flat = tensor[idx].reshape(-1).float()
        finite = torch.isfinite(flat)
        nan_count = torch.isnan(flat).sum().item()
        inf_count = torch.isinf(flat).sum().item()
        safe = flat[finite]
        if safe.numel() == 0:
            return None
        quantiles = torch.quantile(
            safe, torch.tensor([0.05, 0.5, 0.95, 0.99], device=safe.device)
        )
        stats = {
            "min": float(safe.min().item()),
            "max": float(safe.max().item()),
            "p5": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p95": float(quantiles[2].item()),
            "p99": float(quantiles[3].item()),
            "mean": float(safe.mean().item()),
            "std": float(safe.std(unbiased=False).item()),
            "nan": int(nan_count),
            "inf": int(inf_count),
            "gt_3_ratio": float((flat > 3).float().mean().item()),
            "gt_6_ratio": float((flat > 6).float().mean().item()),
            "gt_10_ratio": float((flat > 10).float().mean().item()),
        }
        return stats

    def _debug_scale_stats(self, tag: str, tensor: torch.Tensor, iteration=None, force: bool = False, check_assert: bool = False, sample: int = 8192):
        """
        Debug-only scale statistics; prints single-line JSON when DEBUG_SCALE=1.
        """
        if not self._debug_scale_enabled():
            return None
        if (iteration is not None) and (not force) and (iteration % 500 != 0):
            return None
        base_stats = self._sample_scale_stats(tensor, sample=sample)
        lin_stats = None
        if tensor is not None:
            lin_tensor = torch.exp(torch.clamp(tensor, -15.0, 10.0))
            lin_stats = self._sample_scale_stats(lin_tensor, sample=sample)
        radii_stats = None
        if hasattr(self, "max_radii2D") and isinstance(self.max_radii2D, torch.Tensor) and self.max_radii2D.numel() > 0:
            radii_stats = self._sample_scale_stats(self.max_radii2D, sample=sample)
        payload = {"tag": tag, "iter": iteration}
        if base_stats:
            payload.update({
                "log_scale_p50": base_stats.get("p50"),
                "log_scale_p95": base_stats.get("p95"),
                "log_scale_p99": base_stats.get("p99"),
            })
        if lin_stats:
            payload.update({
                "scale_p50": lin_stats.get("p50"),
                "scale_p95": lin_stats.get("p95"),
                "scale_p99": lin_stats.get("p99"),
            })
        if radii_stats:
            payload.update({
                "radii2d_p50": radii_stats.get("p50"),
                "radii2d_p95": radii_stats.get("p95"),
                "radii2d_p99": radii_stats.get("p99"),
            })
        print(json.dumps(payload, default=float))
        if check_assert and base_stats and base_stats.get("p99") is not None and base_stats["p99"] > 6.0:
            raise RuntimeError(f"[DEBUG_SCALE] stage={tag} iter={iteration} log_scale_p99={base_stats['p99']:.3f} > 6 (scene/gaussian_model.py)")
        return payload

    # ------------------------------------------------------------------
    # Mass-debug helpers (gated by DEBUG_MASS env flag)
    # ------------------------------------------------------------------
    def _mass_debug_enabled(self) -> bool:
        return bool(self._debug_mass)

    def _ensure_birth_iter(self, default_iter: int = 0):
        """Allocate birth-iteration buffer when DEBUG_MASS=1."""
        if not self._mass_debug_enabled():
            return
        N = self._xyz.shape[0] if isinstance(self._xyz, torch.Tensor) else 0
        device = self._xyz.device if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0 else "cuda"
        if self._birth_iter is None or self._birth_iter.shape[0] != N:
            self._birth_iter = torch.full((N,), int(default_iter), device=device, dtype=torch.int32)

    def _assert_birth_align(self):
        if not self._mass_debug_enabled():
            return
        if self._birth_iter is None:
            return
        if self._birth_iter.shape[0] != self._xyz.shape[0]:
            raise RuntimeError(f"[DEBUG_MASS] birth_iter mismatch: birth {self._birth_iter.shape[0]} vs xyz {self._xyz.shape[0]}")

    def _sample_percentiles(self, tensor: torch.Tensor, sample: int = 8192, quantiles=(0.5, 0.9, 0.95, 0.99)):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        N = tensor.shape[0]
        s = min(sample, N)
        if s <= 0:
            return None
        idx = torch.randint(0, N, (s,), device=tensor.device)
        flat = tensor[idx].reshape(-1).float()
        finite = torch.isfinite(flat)
        if finite.sum() == 0:
            return None
        flat = flat[finite]
        q_tensor = torch.quantile(flat, torch.tensor(list(quantiles), device=flat.device))
        names = {0.5: "p50", 0.9: "p90", 0.95: "p95", 0.99: "p99"}
        stats = {names[q]: float(q_tensor[i].item()) for i, q in enumerate(quantiles)}
        stats["min"] = float(flat.min().item())
        stats["max"] = float(flat.max().item())
        stats["mean"] = float(flat.mean().item())
        return stats

    def _collect_point_stats(self, mask: torch.Tensor = None, sample: int = 8192):
        """Collect opacity/log_scale/scale/radii2D quantiles from mask or full set."""
        if not self._mass_debug_enabled():
            return None
        N = self._xyz.shape[0] if isinstance(self._xyz, torch.Tensor) else 0
        if N == 0:
            return None
        if mask is not None:
            if mask.dtype == torch.bool:
                if mask.shape[0] != N or mask.sum() == 0:
                    return None
                idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
                if idx.numel() == 0:
                    return None
            else:
                idx = mask
                if idx.numel() == 0:
                    return None
                idx = idx.to(device=self._xyz.device, dtype=torch.long)
            s = min(sample, idx.shape[0])
            perm = torch.randperm(idx.shape[0], device=idx.device)[:s]
            idx = idx[perm]
        else:
            s = min(sample, N)
            idx = torch.randint(0, N, (s,), device=self._xyz.device)

        opacity = self.get_opacity.squeeze()[idx] if hasattr(self, "get_opacity") else None
        log_scale = torch.clamp(self._scaling_base[idx], -15.0, 10.0) if isinstance(self._scaling_base, torch.Tensor) and self._scaling_base.numel() > 0 else None
        scale = torch.exp(log_scale) if log_scale is not None else None
        radii = self.max_radii2D[idx] if isinstance(self.max_radii2D, torch.Tensor) and self.max_radii2D.numel() > 0 else None

        return {
            "opacity": self._sample_percentiles(opacity, sample=opacity.numel() if opacity is not None else 0) if opacity is not None else None,
            "log_scale": self._sample_percentiles(log_scale, sample=log_scale.numel() if log_scale is not None else 0) if log_scale is not None else None,
            "scale": self._sample_percentiles(scale, sample=scale.numel() if scale is not None else 0) if scale is not None else None,
            "radii2D": self._sample_percentiles(radii, sample=radii.numel() if radii is not None else 0) if radii is not None else None,
        }

    def _record_densify_slice(self, start: int, end: int, iteration: int):
        if not self._mass_debug_enabled():
            return
        self._pending_densify_slices.append((int(start), int(end), int(iteration) if iteration is not None else 0))

    def _emit_densify_new_stats(self, iteration: int):
        if not self._mass_debug_enabled():
            self._pending_densify_slices = []
            return
        if not self._pending_densify_slices:
            return
        all_stats = self._collect_point_stats(mask=None, sample=8192)
        for start, end, it in self._pending_densify_slices:
            new_count = max(0, end - start)
            if new_count <= 0:
                continue
            idx = torch.arange(start, end, device=self._xyz.device)
            new_stats = self._collect_point_stats(mask=idx, sample=min(8192, new_count))
            payload = {
                "tag": "densify_new_points",
                "iter": int(iteration),
                "new_count": int(new_count),
                "new_stats": new_stats,
                "all_stats": all_stats,
            }
            print(json.dumps(payload, default=float))
        self._pending_densify_slices = []

    def _safe_corr(self, a: torch.Tensor, b: torch.Tensor):
        if a is None or b is None or a.numel() < 2 or b.numel() < 2:
            return None
        a = a.float()
        b = b.float()
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.std(unbiased=False) * b.std(unbiased=False)) + 1e-8
        if torch.any(denom == 0):
            return None
        return float((a * b).mean().item() / denom.mean().item())

    def _log_prune_breakdown(self, iteration: int, prune_mask: torch.Tensor, reason_counts: dict, N_before: int):
        if not self._mass_debug_enabled():
            return
        if prune_mask is None or prune_mask.numel() == 0:
            return
        iter_val = int(iteration) if iteration is not None else 0
        union = int(prune_mask.sum().item())
        payload = {
            "tag": "prune_breakdown",
            "iter": iter_val,
            "N_before": int(N_before),
            "N_after": int(N_before - union),
            "reason": {k: int(v) for k, v in reason_counts.items()},
            "N_union": int(union),
        }
        all_stats = self._collect_point_stats(mask=None, sample=8192)
        pruned_stats = None
        age_stats = None
        if union > 0:
            pruned_stats = self._collect_point_stats(mask=prune_mask, sample=min(8192, union))
            if self._birth_iter is not None and self._birth_iter.shape[0] == prune_mask.shape[0]:
                ages = (iter_val - self._birth_iter[prune_mask].int()).float()
                if ages.numel() > 0:
                    age_stats = self._sample_percentiles(ages, sample=min(8192, ages.numel()), quantiles=(0.5, 0.9, 0.99))
        payload["stats"] = {"all": all_stats, "pruned": pruned_stats}
        if age_stats is not None:
            payload["age"] = age_stats
        print(json.dumps(payload, default=float))

    # ------------------------------------------------------------------
    # Densify/Prune telemetry (lightweight, reset at summary)
    # ------------------------------------------------------------------
    def _init_dp_stats(self):
        self._dp_stats = {
            "densify_calls": 0,
            "clone_cand": 0,
            "split_cand": 0,
            "clone_sel": 0,
            "clone_add": 0,
            "split_sel": 0,
            "split_add": 0,
            "prune_total": 0,
            "prune_low_op": 0,
            "prune_size2d": 0,
            "prune_size3d": 0,
            "prune_mask": 0,
            "prune_capacity": 0,
            "sel_clone_stats": None,
            "sel_split_stats": None,
            "new_clone_stats": None,
            "new_split_stats": None,
            "prune_stats": None,
            "last_prune_iter": None,
            "last_prune_removed": None,
            "last_prune_before": None,
        }
        self._pending_new_slices = []

    def _clear_dp_deltas(self):
        for key in [
            "densify_calls",
            "clone_cand",
            "split_cand",
            "clone_sel",
            "clone_add",
            "split_sel",
            "split_add",
            "prune_total",
            "prune_low_op",
            "prune_size2d",
            "prune_size3d",
            "prune_mask",
            "prune_capacity",
        ]:
            self._dp_stats[key] = 0
        for key in ["sel_clone_stats", "sel_split_stats", "new_clone_stats", "new_split_stats", "prune_stats"]:
            self._dp_stats[key] = None
        self._pending_new_slices = []

    def _sample_indices(self, mask: torch.Tensor, k: int = 4096):
        if mask is None or not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            return None
        if mask.dtype == torch.bool:
            idx = torch.nonzero(mask, as_tuple=False).flatten()
        else:
            idx = mask
        if idx.numel() == 0:
            return None
        if idx.numel() > k:
            perm = torch.randperm(idx.numel(), device=idx.device)[:k]
            idx = idx[perm]
        return idx

    def _extract_sample(self, tensor: torch.Tensor, idx: torch.Tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        try:
            sample = tensor[idx]
        except Exception:
            return None
        sample = sample.reshape(-1).float()
        if sample.numel() == 0:
            return None
        finite = sample[torch.isfinite(sample)]
        if finite.numel() == 0:
            return None
        return finite

    def _quantile_dict(self, prefix: str, values: torch.Tensor, qs=(0.5, 0.95)):
        if values is None or values.numel() == 0:
            return {}
        probs = torch.tensor(qs, device=values.device)
        quantiles = torch.quantile(values, probs)
        stats = {}
        for prob, val in zip(qs, quantiles):
            stats[f"{prefix}_p{int(prob * 100):02d}"] = float(val.item())
        return stats

    def _gather_point_stats(self, mask: torch.Tensor, grad_values: torch.Tensor = None, include_radii: bool = False, max_sample: int = 4096):
        with torch.no_grad():
            idx = self._sample_indices(mask, k=max_sample)
            if idx is None:
                return None
            stats = {}
            if grad_values is not None and isinstance(grad_values, torch.Tensor) and grad_values.numel() > 0:
                grad_tensor = grad_values
                if grad_tensor.dim() > 1:
                    grad_tensor = torch.norm(grad_tensor, dim=-1)
                g_sample = self._extract_sample(grad_tensor, idx)
                stats.update(self._quantile_dict("grad", g_sample, qs=(0.5, 0.95)))

            op_source = self.get_opacity if hasattr(self, "get_opacity") else None
            if isinstance(op_source, torch.Tensor) and op_source.numel() > 0:
                op_sample = self._extract_sample(op_source.squeeze(), idx)
                stats.update(self._quantile_dict("op", op_sample, qs=(0.5, 0.95)))

            log_source = self._scaling_base if isinstance(self._scaling_base, torch.Tensor) else None
            if log_source is not None and log_source.numel() > 0:
                log_vals = torch.max(log_source, dim=1).values if log_source.dim() > 1 else log_source
                log_sample = self._extract_sample(log_vals, idx)
                stats.update(self._quantile_dict("logS", log_sample, qs=(0.5, 0.95)))
                if log_sample is not None:
                    lin_sample = torch.exp(torch.clamp(log_sample, -15.0, 10.0))
                    stats.update(self._quantile_dict("scale_lin", lin_sample, qs=(0.5, 0.95)))

            if include_radii and hasattr(self, "max_radii2D") and isinstance(self.max_radii2D, torch.Tensor):
                r_sample = self._extract_sample(self.max_radii2D, idx)
                stats.update(self._quantile_dict("r2d", r_sample, qs=(0.95, 0.99)))

            return stats if stats else None

    def _record_densify_selection(self, kind: str, selected_mask: torch.Tensor, grads: torch.Tensor = None, multiplier: int = 1):
        if selected_mask is None or not isinstance(selected_mask, torch.Tensor):
            return
        count = int(selected_mask.sum().item()) if selected_mask.numel() > 0 else 0
        if count <= 0:
            return
        if kind == "clone":
            self._dp_stats["clone_sel"] += count
            self._dp_stats["clone_add"] += count * multiplier
            stats_key = "sel_clone_stats"
        else:
            self._dp_stats["split_sel"] += count
            self._dp_stats["split_add"] += count * multiplier
            stats_key = "sel_split_stats"

        stats = self._gather_point_stats(selected_mask, grad_values=grads, include_radii=False)
        self._dp_stats[stats_key] = stats

    def _record_densify_candidates(self, kind: str, mask: torch.Tensor):
        if mask is None or not isinstance(mask, torch.Tensor):
            return
        cand = int(mask.sum().item()) if mask.numel() > 0 else 0
        if kind == "clone":
            self._dp_stats["clone_cand"] += cand
        else:
            self._dp_stats["split_cand"] += cand

    def _record_new_points_slice(self, kind: str, start: int, end: int):
        if kind not in ("clone", "split"):
            return
        if start >= end:
            return
        self._pending_new_slices.append((kind, int(end - start)))  # store length only; start may shift after prune

    def _finalize_new_point_stats(self):
        if not self._pending_new_slices:
            self._pending_new_slices = []
            return
        with torch.no_grad():
            for kind, length in self._pending_new_slices:
                if length <= 0:
                    continue
                end = self._xyz.shape[0]
                start = max(0, end - length)
                if start >= end:
                    continue
                idx = torch.arange(start, end, device=self._xyz.device, dtype=torch.long)
                stats = self._gather_point_stats(idx, grad_values=None, include_radii=True)
                if kind == "clone":
                    self._dp_stats["new_clone_stats"] = stats
                else:
                    self._dp_stats["new_split_stats"] = stats
        self._pending_new_slices = []

    def _record_prune_event(self, prune_mask: torch.Tensor, reason_masks: dict, iteration: int, N_before: int):
        if prune_mask is None or not isinstance(prune_mask, torch.Tensor):
            return
        removed = int(prune_mask.sum().item()) if prune_mask.numel() > 0 else 0
        if removed <= 0:
            return
        # Reason counts are measured on the intersection with prune_mask; masks may overlap.
        local_counts = {
            "low_op": 0,
            "size2d": 0,
            "size3d": 0,
            "mask": 0,
            "capacity": 0,
        }
        if isinstance(reason_masks, dict):
            for reason_key, target in [
                ("low_opacity", "low_op"),
                ("size2d", "size2d"),
                ("size3d", "size3d"),
                ("mask", "mask"),
                ("capacity", "capacity"),
            ]:
                mask_val = reason_masks.get(reason_key)
                if isinstance(mask_val, torch.Tensor) and mask_val.numel() == prune_mask.numel():
                    local_counts[target] = int(torch.logical_and(prune_mask, mask_val).sum().item())

        self._dp_stats["prune_total"] += removed
        self._dp_stats["prune_low_op"] += local_counts["low_op"]
        self._dp_stats["prune_size2d"] += local_counts["size2d"]
        self._dp_stats["prune_size3d"] += local_counts["size3d"]
        self._dp_stats["prune_mask"] += local_counts["mask"]
        self._dp_stats["prune_capacity"] += local_counts["capacity"]

        stats = self._gather_point_stats(prune_mask, grad_values=None, include_radii=True)
        self._dp_stats["prune_stats"] = stats

        self._dp_stats["last_prune_iter"] = int(iteration) if iteration is not None else None
        self._dp_stats["last_prune_removed"] = removed
        self._dp_stats["last_prune_before"] = int(N_before) if N_before is not None else None

        before = max(1, int(N_before) if N_before is not None else prune_mask.shape[0])
        ratio = removed / float(before)
        if ratio >= self._prune_alert_thresh:
            iter_val = int(iteration) if iteration is not None else 0
            reason_str = (
                f"low_op={local_counts['low_op']:,} "
                f"size2d={local_counts['size2d']:,} "
                f"size3d={local_counts['size3d']:,} "
                f"mask={local_counts['mask']:,} "
                f"cap={local_counts['capacity']:,}"
            )
            print(f"[ALERT] Large prune @iter={iter_val}: removed={removed:,} ({ratio * 100:.1f}%) | {reason_str}")

    def consume_densify_prune_stats(self, reset: bool = True):
        if not hasattr(self, "_dp_stats"):
            return {}, {}
        counters = {
            k: self._dp_stats.get(k, 0)
            for k in [
                "densify_calls",
                "clone_cand",
                "split_cand",
                "clone_sel",
                "clone_add",
                "split_sel",
                "split_add",
                "prune_total",
                "prune_low_op",
                "prune_size2d",
                "prune_size3d",
                "prune_mask",
                "prune_capacity",
            ]
        }
        stats = {
            "sel_clone": self._dp_stats.get("sel_clone_stats"),
            "sel_split": self._dp_stats.get("sel_split_stats"),
            "new_clone": self._dp_stats.get("new_clone_stats"),
            "new_split": self._dp_stats.get("new_split_stats"),
            "prune": self._dp_stats.get("prune_stats"),
        }
        if reset:
            self._clear_dp_deltas()
        return counters, stats

    def _safe_log_scale_for_new_points(self, log_s: torch.Tensor, lo: float = -15.0, hi: float = 3.5) -> torch.Tensor:
        """
        Clamp log-scale for newly created Gaussians to prevent runaway propagation.
        GlowGS-only safety layer; LocoGS keeps raw log-scales.
        """
        if log_s is None or log_s.numel() == 0:
            return log_s
        return torch.clamp(log_s, lo, hi)

    def project_scaling_base_(self, lo: float = -15.0, hi: float = 4.0):
        """
        In-place projection to keep existing log-scales in a recoverable range.
        GlowGS-only safety layer (LocoGS has no clamp in get_scaling_base).
        """
        if self._scaling_base is None or self._scaling_base.numel() == 0:
            return
        with torch.no_grad():
            self._scaling_base.clamp_(lo, hi)

    def get_cache_stats(self):
        """Return cache hit/miss statistics for update_attributes."""
        hits = getattr(self, '_cache_hit_count', 0)
        misses = getattr(self, '_cache_miss_count', 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        return {
            'cache_hits': hits,
            'cache_misses': misses,
            'total_calls': total,
            'hit_rate_percent': hit_rate
        }
    
    def reset_cache_stats(self):
        """Reset cache statistics counters."""
        self._cache_hit_count = 0
        self._cache_miss_count = 0
    
    def _stabilize_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Numerical stability for feature tensors (legacy, use clamp directly in hot path)."""
        # Fast path: just clamp (NaN/Inf checks are expensive sync operations)
        return torch.clamp(feats, -10.0, 10.0)

    @property
    def get_scaling_base(self):
        # Clamp before exp to prevent overflow (exp(10) ≈ 22026, exp(20) ≈ 4.8e8)
        clamped_scaling = torch.clamp(self._scaling_base, -15.0, 10.0)
        return self.scaling_base_activation(clamped_scaling)

    @property
    def get_scaling(self):
        return self.get_scaling_base * self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        # Add epsilon to prevent NaN from normalize when rotation is all zeros
        # This can happen during initialization or if gradients zero out the rotation
        rotation = self._rotation
        eps = 1e-6
        norm = torch.norm(rotation, dim=-1, keepdim=True)
        # If norm is too small, use a default rotation (1, 0, 0, 0)
        rotation = torch.where(
            norm > eps,
            rotation,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
        )
        return self.rotation_activation(rotation)
    
    @property
    def get_xyz(self):
        return self.xyz_activation(self._xyz)

    @property
    def get_contracted_xyz(self):
        # Smart cache: track xyz data pointer to detect changes (densify/prune/optimizer)
        # This is much more efficient than shape check alone
        N = self._xyz.shape[0]
        if N == 0:
            return torch.empty((0, 3), device=self._xyz.device)
        
        # Use data pointer + size as cache key (changes on any modification)
        current_key = (self._xyz.data_ptr(), N)
        
        if not hasattr(self, '_cached_contracted_key') or self._cached_contracted_key != current_key:
            # Cache miss - recompute
            with torch.no_grad():  # No grad needed for coordinate transformation
                contracted = contract_to_unisphere(self.get_xyz.detach(), self._aabb)
                # Fast clamp to [-1, 1] (NaN/Inf checks are expensive sync operations, skip in hot path)
                self._cached_contracted_xyz = torch.clamp(contracted, -1.0, 1.0)
                self._cached_contracted_key = current_key
        
        return self._cached_contracted_xyz
        
    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.max_sh_degree > 0:
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_mask(self):
        return self.mask_activation(self._mask)

    @property
    def get_sh_mask(self):
        return self.sh_mask_activation(self._sh_mask)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        if is_verbose():
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        # explicit attributes
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        # Initialize SH mask high so sigmoid(6.0)~0.997, allowing full specular details from start
        self._sh_mask = nn.Parameter(torch.full((fused_point_cloud.shape[0], self.max_sh_degree), 6.0, device="cuda").requires_grad_(True))

        # _features_rest is always implicit (generated by encoder)
        # Initialize to empty tensor (will be populated by update_attributes)
        self._features_rest = torch.empty(0)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._ensure_birth_iter(default_iter=0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Store densify schedule for debug gating
        self._densify_from_iter = getattr(training_args, 'densify_from_iter', 0)
        self._densification_interval = getattr(training_args, 'densification_interval', 100)
        self._ensure_birth_iter(default_iter=0)
        
        # Update capacity config from training_args (allow CLI override)
        max_gaussians = getattr(training_args, 'max_gaussians', 6_000_000)
        densify_until = getattr(training_args, 'densify_until_iter', 15000)
        prune_interval = getattr(training_args, 'prune_interval', 1000)
        
        # Store warmup/ramp schedule from training args
        self.mass_aware_start_iter = getattr(training_args, 'mass_aware_start_iter', 3000)
        self.mass_aware_end_iter = getattr(training_args, 'mass_aware_end_iter', 5000)
        self.size_prune_start_iter = getattr(training_args, 'size_prune_start_iter', 4000)
        self.size_prune_end_iter = getattr(training_args, 'size_prune_end_iter', 6000)
        
        self.gaussian_capacity_config.update({
            "max_point_count": max_gaussians,
            "densify_until_iter": densify_until,
            "prune_interval": prune_interval,
        })

        param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_dc_lr, "name": "f_dc"},
            {'params': [self._scaling_base], 'lr': training_args.scaling_base_lr, "name": "scaling_base"},
            {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
            {'params': [self._sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"}
        ]
        
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        param_groups_i = [
            {'params': self._grid.parameters(), 'lr': 0.0, "name": "grid"},
            {'params': list(self._opacity_head.parameters()), 'lr': 0.0, "name": "opacity"},
            {'params': self._features_rest_head.parameters(), 'lr': 0.0, "name": "f_rest"},
            {'params': self._scaling_head.parameters(), 'lr': 0.0, "name": "scaling"},
            {'params': list(self._rotation_head.parameters()), 'lr': 0.0, "name": "rotation"},
        ]
        # Include projection layers into implicit optimizer so they can adapt
        try:
            proj_params = []
            if hasattr(self, '_geometry_input_proj') and self._geometry_input_proj is not None:
                proj_params.extend(list(self._geometry_input_proj.parameters()))
            if hasattr(self, '_appearance_input_proj') and self._appearance_input_proj is not None:
                proj_params.extend(list(self._appearance_input_proj.parameters()))
            # Fallback: legacy projection layers
            if len(proj_params) == 0:
                if hasattr(self, '_opacity_input_proj'):
                    proj_params.extend(list(self._opacity_input_proj.parameters()))
                if hasattr(self, '_sh_input_proj'):
                    proj_params.extend(list(self._sh_input_proj.parameters()))
            if len(proj_params) > 0:
                param_groups_i.append({'params': proj_params, 'lr': 0.0, "name": "proj"})
        except Exception:
            pass
        self.optimizer_i = torch.optim.Adam(param_groups_i, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init,
                                                     lr_final=training_args.grid_lr_final,
                                                     lr_delay_steps=training_args.grid_lr_delay_steps,
                                                     lr_delay_mult=training_args.grid_lr_delay_mult,
                                                     max_steps=training_args.grid_lr_max_steps)

        self.opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_lr_init,
                                                        lr_final=training_args.opacity_lr_final,
                                                        lr_delay_steps=training_args.opacity_lr_delay_steps,
                                                        lr_delay_mult=training_args.opacity_lr_delay_mult,
                                                        max_steps=training_args.opacity_lr_max_steps)

        self.feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_rest_lr_init,
                                                        lr_final=training_args.feature_rest_lr_final,
                                                        lr_delay_steps=training_args.feature_rest_lr_delay_steps,
                                                        lr_delay_mult=training_args.feature_rest_lr_delay_mult,
                                                        max_steps=training_args.feature_rest_lr_max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init,
                                                        lr_final=training_args.scaling_lr_final,
                                                        lr_delay_steps=training_args.scaling_lr_delay_steps,
                                                        lr_delay_mult=training_args.scaling_lr_delay_mult,
                                                        max_steps=training_args.scaling_lr_max_steps)

        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init,
                                                         lr_final=training_args.rotation_lr_final,
                                                         lr_delay_steps=training_args.rotation_lr_delay_steps,
                                                         lr_delay_mult=training_args.rotation_lr_delay_mult,
                                                         max_steps=training_args.rotation_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_rest":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
        
        for param_group in self.optimizer_i.param_groups:
            if param_group["name"] == "grid":
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "opacity":
                lr = self.opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_dc":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_rest":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = torch.clamp_min(self.scaling_base_inverse_activation(self.get_scaling), min=-20).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def load_ply(self, path):
    #     plydata = PlyData.read(path)

    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])),  axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    #     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    #     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     sh_mask = np.stack([np.asarray(plydata.elements[0][f"sh_mask_{i}"]) for i in range(3)],  axis=1)

    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._sh_mask = nn.Parameter(torch.tensor(sh_mask, dtype=torch.float, device="cuda").requires_grad_(True))

    #     self.active_sh_degree = self.max_sh_degree

    # ------------------------------------------------------------------
    # VM upsample → Optimizer state sync
    # ------------------------------------------------------------------
    def _register_vm_optimizer(self):
        """
        After GeoEncoder.upsample_resolution(), the 6 VM nn.Parameters have
        been replaced via setattr with new (larger) tensors.  Adam still holds
        references to the OLD tensors, so we must:

          1. Replace the 'grid' param list in optimizer_i with the current
             self._grid.parameters().
          2. Delete stale Adam states (exp_avg / exp_avg_sq) for the dead
             old tensors.
          3. New parameters will get fresh Adam states on their next .step().

        This is the "精细操作" approach — unchanged parameters (hash encoder,
        FiLM MLP, projection) keep their accumulated momentum.
        """
        if self.optimizer_i is None:
            return

        new_params = list(self._grid.parameters())
        new_id_set = {id(p) for p in new_params}

        replaced = 0
        for group in self.optimizer_i.param_groups:
            if group["name"] == "grid":
                old_params = group["params"]
                # Delete optimizer state for parameters that no longer exist
                for old_p in old_params:
                    if id(old_p) not in new_id_set:
                        if old_p in self.optimizer_i.state:
                            del self.optimizer_i.state[old_p]
                        replaced += 1
                # Replace param list with current module parameters
                group["params"] = new_params
                break

        print(f"[OPTIMIZER] VM upsample sync: replaced {replaced} stale param "
              f"refs, optimizer_i 'grid' group now has {len(new_params)} params")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iteration=None):
        """
        Prune Gaussians based on boolean mask.
        
        Args:
            mask: Boolean tensor where True indicates points to REMOVE
        """
        # Ensure mask is on CUDA for consistency
        if not mask.is_cuda:
            mask = mask.cuda()
        
        valid_points_mask = ~mask
        
        if self.optimizer is None:
            # ============================================
            # INFERENCE MODE: No optimizer, manual pruning
            # ============================================
            def safe_prune(tensor, mask, name=""):
                """Safely prune a tensor/parameter, handling all edge cases."""
                if tensor is None:
                    return tensor
                if not isinstance(tensor, (torch.Tensor, nn.Parameter)):
                    return tensor
                if tensor.numel() == 0:
                    return tensor
                # Check shape compatibility
                if len(tensor.shape) == 0 or tensor.shape[0] != mask.shape[0]:
                    # Shape mismatch - skip this tensor
                    return tensor
                
                # Ensure device match
                mask_device = mask
                if tensor.is_cuda and not mask.is_cuda:
                    mask_device = mask.cuda()
                elif not tensor.is_cuda and mask.is_cuda:
                    mask_device = mask.cpu()
                
                # Perform slicing
                pruned = tensor[mask_device]
                
                # Preserve type (Parameter vs Tensor)
                if isinstance(tensor, nn.Parameter):
                    return nn.Parameter(pruned)
                return pruned
            
            # Prune core attributes
            self._xyz = safe_prune(self._xyz, valid_points_mask, "xyz")
            self._features_dc = safe_prune(self._features_dc, valid_points_mask, "f_dc")
            self._scaling_base = safe_prune(self._scaling_base, valid_points_mask, "scaling_base")
            self._mask = safe_prune(self._mask, valid_points_mask, "mask")
            self._sh_mask = safe_prune(self._sh_mask, valid_points_mask, "sh_mask")
            
            # Prune f_rest if it exists and matches shape
            if isinstance(self._features_rest, torch.Tensor) and self._features_rest.numel() > 0:
                if self._features_rest.shape[0] == mask.shape[0]:
                    self._features_rest = safe_prune(self._features_rest, valid_points_mask, "f_rest")
            
            # Prune computed attributes (from update_attributes)
            for attr_name in ['_opacity', '_scaling', '_rotation']:
                attr = getattr(self, attr_name, None)
                if isinstance(attr, torch.Tensor) and attr.numel() > 0 and attr.shape[0] == mask.shape[0]:
                    setattr(self, attr_name, safe_prune(attr, valid_points_mask, attr_name))
            
            # Prune precomputed cache
            if getattr(self, '_precomputed_cache', None) is not None:
                for k in list(self._precomputed_cache.keys()):
                    v = self._precomputed_cache[k]
                    if isinstance(v, torch.Tensor) and v.numel() > 0 and v.shape[0] == mask.shape[0]:
                        self._precomputed_cache[k] = safe_prune(v, valid_points_mask, f"cache_{k}")
            
            # Prune training buffers ONLY if they exist and are initialized
            # In pure inference mode, these might be uninitialized or on wrong device
            if hasattr(self, 'xyz_gradient_accum') and isinstance(self.xyz_gradient_accum, torch.Tensor):
                if self.xyz_gradient_accum.numel() > 0 and self.xyz_gradient_accum.shape[0] == mask.shape[0]:
                    self.xyz_gradient_accum = safe_prune(self.xyz_gradient_accum, valid_points_mask, "grad_accum")
            
            if hasattr(self, 'denom') and isinstance(self.denom, torch.Tensor):
                if self.denom.numel() > 0 and self.denom.shape[0] == mask.shape[0]:
                    self.denom = safe_prune(self.denom, valid_points_mask, "denom")
            
            if hasattr(self, 'max_radii2D') and isinstance(self.max_radii2D, torch.Tensor):
                if self.max_radii2D.numel() > 0 and self.max_radii2D.shape[0] == mask.shape[0]:
                    self.max_radii2D = safe_prune(self.max_radii2D, valid_points_mask, "max_radii2D")
            
        else:
            # ============================================
            # TRAINING MODE: Use optimizer-aware pruning
            # ============================================
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._scaling_base = optimizable_tensors["scaling_base"]
            self._mask = optimizable_tensors["mask"]
            self._sh_mask = optimizable_tensors["sh_mask"]

            # Training buffers - these MUST exist and match in training mode
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self._mass_debug_enabled() and isinstance(self._birth_iter, torch.Tensor) and self._birth_iter.numel() > 0:
            if self._birth_iter.shape[0] == valid_points_mask.shape[0]:
                self._birth_iter = self._birth_iter[valid_points_mask]
            else:
                # Shape drift shouldn't happen; realign defensively
                self._ensure_birth_iter(default_iter=0)
                if self._birth_iter.shape[0] == valid_points_mask.shape[0]:
                    self._birth_iter = self._birth_iter[valid_points_mask]

        # Clear cached contracted_xyz since xyz changed
        if hasattr(self, '_cached_contracted_xyz'):
            delattr(self, '_cached_contracted_xyz')

        self._debug_scale_stats(tag="after_prune_points", tensor=self._scaling_base, iteration=iteration, force=True, check_assert=True)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest=None, iteration=None, kind: str = None):
        n_before = self._xyz.shape[0]
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "scaling_base": new_scaling_base,
            "mask": new_mask,
            "sh_mask": new_sh_mask,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling_base = optimizable_tensors["scaling_base"]
        self._mask = optimizable_tensors["mask"]
        self._sh_mask = optimizable_tensors["sh_mask"]

        n_after = self.get_xyz.shape[0]
        added = n_after - n_before
        if added > 0:
            self._record_new_points_slice(kind=kind, start=n_after - added, end=n_after)

        if self._mass_debug_enabled():
            self._ensure_birth_iter(default_iter=0)
            if added > 0:
                birth_val = int(iteration) if iteration is not None else 0
                new_birth = torch.full((added,), birth_val, device=self._xyz.device, dtype=torch.int32)
                self._birth_iter = torch.cat((self._birth_iter, new_birth), dim=0)
                self._assert_birth_align()
                self._record_densify_slice(n_before, n_after, birth_val)

        self.xyz_gradient_accum = torch.zeros((n_after, 1), device="cuda")
        self.denom = torch.zeros((n_after, 1), device="cuda")
        self.max_radii2D = torch.zeros((n_after,), device="cuda")
        
        # Invalidate attribute caches; shapes changed
        self._cached_attributes_hash = None
        if hasattr(self, '_cached_contracted_key'):
            self._cached_contracted_key = None

        if self._debug_scale_enabled() and n_after > n_before:
            added = n_after - n_before
            new_slice = slice(n_after - added, n_after)
            new_stats = self._sample_scale_stats(self._scaling_base[new_slice], sample=min(2048, added))
            old_stats = self._sample_scale_stats(self._scaling_base[:n_before], sample=min(2048, n_before)) if n_before > 0 else None
            payload = {
                "stage": "after_densify_postfix",
                "iter": iteration,
                "added": int(added),
                "scale_new": new_stats,
                "scale_prev": old_stats,
            }
            print(json.dumps(payload, default=float))
            if new_stats and old_stats and new_stats.get("p50") is not None and old_stats.get("p95") is not None:
                if new_stats["p50"] > old_stats["p95"]:
                    raise RuntimeError(f"[DEBUG_SCALE] iter={iteration} new_points scale_base p50 {new_stats['p50']:.3f} exceeds prev p95 {old_stats['p95']:.3f} (densification_postfix)")
        self._debug_scale_stats(tag="after_densify_postfix", tensor=self._scaling_base, iteration=iteration, force=True, check_assert=True)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, iteration=None):
        """
        Split large Gaussians with high gradient.
        Uses original 3DGS logic with FPS guard for screen-space size.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        grad_mask = padded_grad >= grad_threshold
        self._record_densify_candidates(kind="split", mask=grad_mask)

        selected_pts_mask = grad_mask
        
        # World-space size constraint (original 3DGS)
        is_large_ws = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_large_ws)

        num_sel = int(selected_pts_mask.sum().item())
        if num_sel == 0:
            return
        # Telemetry: split selection (adds N new points per source)
        self._record_densify_selection(kind="split", selected_mask=selected_pts_mask, grads=padded_grad, multiplier=N)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rot_src = self._rotation
        if rot_src is None or (isinstance(rot_src, torch.Tensor) and rot_src.numel() == 0):
            if is_verbose():
                print("[densify_and_split] rotation buffer empty, using identity quaternions")
            rot_src = self._rotation_init.expand(self.get_xyz.shape[0], -1).to(device=self._xyz.device, dtype=self._xyz.dtype)
        rot_selected = rot_src[selected_pts_mask]
        if rot_selected.numel() == 0:
            return
        rots = build_rotation(rot_selected).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # Safe log-scale path (GlowGS-only safety)
        safe_log = self._safe_log_scale_for_new_points(self._scaling_base.detach())
        selected_log = safe_log[selected_pts_mask].repeat(N, 1)
        selected_lin = torch.exp(selected_log)
        shrunk_lin = selected_lin / (0.8 * N)
        new_log = torch.log(shrunk_lin.clamp_min(1e-6))
        new_scaling_base = self._safe_log_scale_for_new_points(new_log)
        new_mask = self._mask[selected_pts_mask].repeat(N,1)
        new_sh_mask = self._sh_mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, iteration=iteration, kind="split")

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iteration=iteration)
        self._debug_scale_stats(tag="after_densify_split", tensor=self._scaling_base, iteration=iteration, force=True, check_assert=True)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iteration=None):
        """
        Clone small Gaussians with high gradient.
        Uses original 3DGS logic with FPS guard for screen-space size.
        """
        grad_norm = torch.norm(grads, dim=-1)
        grad_mask = torch.where(grad_norm >= grad_threshold, True, False)
        self._record_densify_candidates(kind="clone", mask=grad_mask)

        selected_pts_mask = grad_mask
        
        # World-space size constraint (original 3DGS / LocoGS)
        # Clone only small Gaussians (large ones should split instead)
        is_small_ws = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_small_ws)
        
        # [REMOVED] Screen-space size constraint (max_radii2D > 2.0)
        # Reason: This filter blocked sub-pixel Gaussians from cloning, preventing
        # high-frequency texture recovery. Inria 3DGS has no such constraint.

        # Mass Gate: block (large && transparent) Gaussians from cloning
        # Prevents "garbage duplication" that occludes fine details
        # Only affects clone selection, NOT pruning (safe: cannot cause point collapse)
        enable_mass_gate = getattr(self, '_enable_mass_gate', True)
        if enable_mass_gate:
            opacity = self.get_opacity.squeeze()
            opacity_thresh = getattr(self, '_mass_gate_opacity_threshold', 0.3)
            radius_pct = getattr(self, '_mass_gate_radius_percentile', 80.0)
            
            is_transparent = opacity < opacity_thresh
            radii_valid = self.max_radii2D > 0
            if radii_valid.sum() > 0:
                radius_thresh = torch.quantile(
                    self.max_radii2D[radii_valid].float(), 
                    radius_pct / 100.0
                )
                is_large = self.max_radii2D > radius_thresh
                mass_gate_reject = torch.logical_and(is_large, is_transparent)
                selected_pts_mask = torch.logical_and(selected_pts_mask, ~mass_gate_reject)

        num_sel = int(selected_pts_mask.sum().item()) if selected_pts_mask.numel() > 0 else 0
        if num_sel == 0:
            return
        # Telemetry: clone selection (adds one point per source)
        self._record_densify_selection(kind="clone", selected_mask=selected_pts_mask, grads=grads, multiplier=1)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        safe_log = self._safe_log_scale_for_new_points(self._scaling_base.detach())
        new_scaling_base = safe_log[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_sh_mask = self._sh_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, iteration=iteration, kind="clone")
        self.update_attributes(force_update=True, iteration=iteration)
        self._debug_scale_stats(tag="after_densify_clone", tensor=self._scaling_base, iteration=iteration, force=True, check_assert=True)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration=0, mask_prune_threshold=0.01):
        """
        Core densification and pruning logic.
        Simplified to use original 3DGS approach with opacity-based pruning.
        """
        self._dp_stats["densify_calls"] += 1
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Mass-aware bypass: when enable_mass_aware=False, skip size-decay weighting
        # and use raw gradients for both clone and split (standard 3DGS behavior).
        if not self._enable_mass_aware:
            clone_grads = grads
            split_grads = grads
        else:
            warmup_iter = 3000
            tau = 0.5  # FIXED: Increased from 0.1 to reduce over-suppression (safe baseline)

            if iteration < warmup_iter:
                clone_grads = grads
                split_grads = grads
            else:
                # Post-warmup: apply size-decay weight to clone, keep split raw
                # CRITICAL FIX: Use world-space scaling, NOT screen-space max_radii2D
                world_scale = self.get_scaling.max(dim=1).values  # [N] World-space Gaussian radius
                extent_safe = max(extent, 1e-6)
                decay = (world_scale / (tau * extent_safe)) ** 2
                weight = 1.0 / (1.0 + decay)  # [N] Size-decay weight (0.0~1.0)
                
                raw_grad = grads.squeeze(-1)  # [N]
                clone_grads = (raw_grad * weight).unsqueeze(1)
                split_grads = raw_grad.unsqueeze(1)

        self.update_attributes(force_update=True, iteration=iteration)
        self.densify_and_clone(clone_grads, max_grad, extent, iteration=iteration)
        # Refresh attributes after clone (match LocoGS call sequence)
        # Ensures newly cloned points have correct encoder outputs before split decision
        self.update_attributes(force_update=True, iteration=iteration)
        self.densify_and_split(split_grads, max_grad, extent, iteration=iteration)
        self.update_attributes(force_update=True, iteration=iteration)
        # After attributes refresh, gather new-point stats for the slices recorded during densification
        self._finalize_new_point_stats()
        self._emit_densify_new_stats(iteration=iteration)
        
        # === Pruning: Opacity + Mask + Size ===
        N_before = self.get_xyz.shape[0]
        opacity = self.get_opacity.squeeze()
        
        # Base pruning: extremely low opacity (original 3DGS)
        opacity_mask = (opacity < min_opacity)
        prune_mask = opacity_mask
        # Early mask prune (match LocoGS behavior): remove points with very low mask values
        mask_mask = (self.get_mask <= mask_prune_threshold).squeeze()
        prune_mask = torch.logical_or(prune_mask, mask_mask)

        # [RESTORED] Size-based pruning (original 3DGS / LocoGS)
        # Reference: LocoGS/scene/gaussian_model.py:553-558
        # Prune Gaussians that are too large in screen-space or world-space
        # This prevents "blob" accumulation that occludes fine details
        size2d_mask = None
        size3d_mask = None
        if max_screen_size is not None:
            # Screen-space size pruning: remove Gaussians > max_screen_size pixels
            size2d_mask = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, size2d_mask)
            
            # World-space size pruning: remove Gaussians > 0.1 * scene_extent
            # This catches large Gaussians that might not be visible in current view
            size3d_mask = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, size3d_mask)

        if self._mass_debug_enabled():
            reason_counts = {
                "opacity": int(opacity_mask.sum().item()),
                "size": 0,
                "mask": 0,
                "capacity": 0,
            }
            self._log_prune_breakdown(iteration, prune_mask, reason_counts, N_before=N_before)
        # Telemetry: record prune event (reason masks may overlap; counts are per-mask intersections)
            self._record_prune_event(
            prune_mask=prune_mask,
            reason_masks={
                "low_opacity": opacity_mask,
                "size2d": size2d_mask,
                "size3d": size3d_mask,
                "mask": mask_mask,
                "capacity": None,
            },
            iteration=iteration,
            N_before=N_before,
        )
        
        self.prune_points(prune_mask, iteration=iteration)
        self._debug_scale_stats(tag="after_prune", tensor=self._scaling_base, iteration=iteration, force=True, check_assert=True)
        torch.cuda.empty_cache()

    @staticmethod
    def _compute_ramp_weight(iteration, start_iter, end_iter):
        """Compute linear ramp weight: 0 at start, 1 at end."""
        return max(0.0, min(1.0, (iteration - start_iter) / max(1, end_iter - start_iter)))

    def maybe_densify_and_prune(
        self,
        iteration: int,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size,
        mask_prune_threshold: float = 0.005,
    ):
        """
        Capacity-controlled densification wrapper.
        
        Design:
        - Enforces hard point budget (N_max) to prevent memory explosion.
        - Densification only proceeds if: (1) within schedule, (2) under capacity.
        - Post-densification: periodic mask pruning to remove low-contribution Gaussians.
        
        Args:
            iteration: Current training iteration
            max_grad: Gradient threshold for densification
            min_opacity: Minimum opacity for pruning
            extent: Scene extent for scale threshold
            max_screen_size: Max screen-space size for pruning
            mask_prune_threshold: Mask threshold for post-densification pruning
        """
        config = self.gaussian_capacity_config
        max_points = config["max_point_count"]
        densify_until = config["densify_until_iter"]
        prune_interval = config["prune_interval"]
        
        num_points = self.get_xyz.shape[0]
        
        # Dynamic threshold: raise threshold as we approach capacity
        # This naturally slows down point growth when resources are tight
        capacity_ratio = num_points / max_points
        if capacity_ratio > 0.5:
            # Scale threshold: at 50% capacity -> 1x, at 100% capacity -> 3x
            threshold_scale = 1.0 + 2.0 * (capacity_ratio - 0.5) / 0.5
            adaptive_grad = max_grad * threshold_scale
        else:
            adaptive_grad = max_grad
        
        # Densify only if under capacity and within schedule
        if iteration < densify_until and num_points < max_points:
            self.densify_and_prune(adaptive_grad, min_opacity, extent, max_screen_size, iteration=iteration, mask_prune_threshold=mask_prune_threshold)
            new_count = self.get_xyz.shape[0]
            # Log capacity warning if approaching limit
            if new_count >= max_points * 0.9:
                print(f"[CAPACITY] Approaching limit: {new_count}/{max_points} ({new_count/max_points*100:.1f}%)")
        elif iteration >= densify_until:
            # Post-densification phase: periodic mask pruning only
            if iteration % prune_interval == 0:
                self.mask_prune(mask_threshold=mask_prune_threshold, iteration=iteration)
        else:
            # At or over capacity: skip densification, still allow pruning
            if iteration % prune_interval == 0:
                self.mask_prune(mask_threshold=mask_prune_threshold, iteration=iteration)
                print(f"[CAPACITY] At limit ({num_points}>={max_points}), densify skipped, prune only")

    def mask_prune(self, mask_threshold=0.01, iteration=None):
        # Pruning: points with mask <= threshold are removed (LocoGS: 0.01)
        prune_mask = (self.get_mask <= mask_threshold).squeeze()
        N_before = self.get_xyz.shape[0]
        if self._mass_debug_enabled():
            reason_counts = {
                "opacity": 0,
                "size": 0,
                "mask": int(prune_mask.sum().item()),
                "capacity": 0,
            }
            self._log_prune_breakdown(iteration, prune_mask, reason_counts, N_before=N_before)
        # Telemetry: mask-based prune (reason masks may overlap; counts are per-mask intersections)
        self._record_prune_event(
            prune_mask=prune_mask,
            reason_masks={
                "low_opacity": None,
                "size2d": None,
                "size3d": None,
                "mask": prune_mask,
                "capacity": None,
            },
            iteration=iteration,
            N_before=N_before,
        )
        self.prune_points(prune_mask, iteration=iteration)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opacity=None, radii=None, iteration=0):
        """
        Accumulate RAW gradients for densification decision.
        
        Per copilot-instructions.md:
        "Mass-aware is a resource allocation signal for Gaussian budget management 
        (gate/rank/prune), while densify trigger stats must stay RAW (unweighted)."
        
        Mass-Aware Gate is applied in densify_and_clone/split, NOT here.
        
        Args:
            viewspace_point_tensor: Tensor with .grad containing screen-space gradients
            update_filter: Boolean mask for visible points
            opacity: Unused (kept for API compatibility)
            radii: Unused (kept for API compatibility)
            iteration: Unused (kept for API compatibility)
        """
        # RAW gradient accumulation (LocoGS / 3DGS standard behavior)
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1

    def sort_attributes(self):
        xyz = self.get_xyz.float().detach().cpu().numpy()
        xyz_q = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0)) * (2 ** 21 - 1)
        order = mortonEncode(torch.tensor(xyz_q).long()).sort().indices

        self._xyz = self._xyz[order]
        self._scaling_base = self._scaling_base[order]
        self._features_dc = self._features_dc[order]
        self._sh_mask = self._sh_mask[order]

    def compress_attributes(self, path):
        os.makedirs(path, exist_ok=True)

        self.mask_prune()
        self.sort_attributes()

        xyz = self.get_xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()
        scale_base = self._scaling_base.detach().cpu().numpy()
        sh_mask = self.get_sh_mask.detach().cpu().numpy()
        
        # Save model configuration for correct loading
        config = {
            'feature_role_split': self._feature_role_split,
            'geo_resolution': self._geo_resolution,
            'geo_rank': self._geo_rank,
            'geo_channels': self._geo_channels,
            'max_sh_degree': self.max_sh_degree,
            'enable_vm': self._enable_vm,
            'enable_mass_aware': self._enable_mass_aware,
        }
        np.savez(os.path.join(path, "config.npz"), **config)
        
        # Save the complete GeometryAppearanceEncoder state_dict
        # This includes hash_encoder, geo_encoder, and fusion_layer parameters
        # All files use underscore naming format for consistency
        
        grid_state_dict = self._grid.state_dict()
        # Pack all grid components into a single compressed archive to reduce header
        # overhead and improve compression across parameters.
        grid_save_dict = {}
        for key, value in grid_state_dict.items():
            normalized_key = key.replace('.', '_')  # e.g., "hash_encoder.params" -> "hash_encoder_params"
            param_data = value.detach().cpu().numpy()
            # Quantize grid parameters (log quantization) to uint8 (legacy path)
            param_quant, param_scale, param_min = quantize(param_data, bit=6, log=True)
            grid_save_dict[f"{normalized_key}_data"] = param_quant.astype(np.uint8)
            grid_save_dict[f"{normalized_key}_scale"] = param_scale
            grid_save_dict[f"{normalized_key}_min"] = param_min
            grid_save_dict[f"{normalized_key}_orig"] = np.array(key)

        # Save single combined file
        np.savez_compressed(os.path.join(path, "grid_all.npz"), **grid_save_dict)

        # G-PCC encoding (geometry stays as legacy GPCC path)
        encode_xyz(xyz, path)

        # -------- Quality-first artifacts (preferred for rendering fidelity) --------
        # DC in FP16 to preserve SH0/base color fidelity.
        _save_npz(os.path.join(path, "f_dc_fp16.npz"), data=_to_np_fp16(self._features_dc))
        # Log-scale base stored in FP16 to avoid exp-amplified quantization error.
        _save_npz(os.path.join(path, "scale_base_fp16.npz"), data=_to_np_fp16(self._scaling_base))
        # Mask / SH mask stored as logits (no binarization) to keep activation semantics.
        _save_npz(os.path.join(path, "mask_logits_fp16.npz"), data=_to_np_fp16(self._mask))
        _save_npz(os.path.join(path, "sh_mask_logits_fp16.npz"), data=_to_np_fp16(self._sh_mask))
        # Tri-plane VM parameters stored in FP16 to avoid implicit attribute drift.
        if hasattr(self._grid, 'plane_xy'):
            _save_npz(os.path.join(path, "vm_planes_fp16.npz"),
                      plane_xy=_to_np_fp16(self._grid.plane_xy),
                      plane_xz=_to_np_fp16(self._grid.plane_xz),
                      plane_yz=_to_np_fp16(self._grid.plane_yz))
        # Tinycudann heads stored FP16, no quantization.
        _save_npz(os.path.join(path, "mlps_fp16.npz"),
                  opacity=_to_np_fp16(self._opacity_head.state_dict()['params']),
                  scaling=_to_np_fp16(self._scaling_head.state_dict()['params']),
                  rotation=_to_np_fp16(self._rotation_head.state_dict()['params']),
                  features_rest=_to_np_fp16(self._features_rest_head.state_dict()['params']),
                  opacity_numel=np.array([self._opacity_head.state_dict()['params'].numel()]),
                  scaling_numel=np.array([self._scaling_head.state_dict()['params'].numel()]),
                  rotation_numel=np.array([self._rotation_head.state_dict()['params'].numel()]),
                  features_rest_numel=np.array([self._features_rest_head.state_dict()['params'].numel()]))

        # -------- Legacy quantized artifacts (kept for backward compatibility) --------
        f_dc_quant, f_dc_scale, f_dc_min = quantize(f_dc, bit=8)
        scaling_base_quant, scaling_base_scale, scaling_base_min = quantize(scale_base, bit=8)
        np.savez_compressed(os.path.join(path, "f_dc.npz"), data=f_dc_quant.astype(np.uint8), scale=f_dc_scale, min=f_dc_min)
        np.savez_compressed(os.path.join(path, "scale_base.npz"), data=scaling_base_quant.astype(np.uint8), scale=scaling_base_scale, min=scaling_base_min)
        sh_mask_legacy = (sh_mask > 0.01).cumprod(axis=-1).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "sh_mask.npz"), data=sh_mask_legacy)
        mask_legacy = (self.get_mask.detach().cpu().numpy() > 0.01).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "mask.npz"), data=mask_legacy)
        mlp_save = {}
        opacity_params = self._opacity_head.state_dict()['params'].cpu().detach().numpy()
        f_rest_params = self._features_rest_head.state_dict()['params'].cpu().detach().numpy()
        scale_params = self._scaling_head.state_dict()['params'].cpu().detach().numpy()
        rotation_params = self._rotation_head.state_dict()['params'].cpu().detach().numpy()
        o_q, o_s, o_m = quantize(opacity_params, bit=8, log=True)
        fr_q, fr_s, fr_m = quantize(f_rest_params, bit=8, log=True)
        sc_q, sc_s, sc_m = quantize(scale_params, bit=8, log=True)
        rt_q, rt_s, rt_m = quantize(rotation_params, bit=8, log=True)
        mlp_save['opacity_data'] = o_q.astype(np.uint8)
        mlp_save['opacity_scale'] = o_s
        mlp_save['opacity_min'] = o_m
        mlp_save['f_rest_data'] = fr_q.astype(np.uint8)
        mlp_save['f_rest_scale'] = fr_s
        mlp_save['f_rest_min'] = fr_m
        mlp_save['scale_data'] = sc_q.astype(np.uint8)
        mlp_save['scale_scale'] = sc_s
        mlp_save['scale_min'] = sc_m
        mlp_save['rotation_data'] = rt_q.astype(np.uint8)
        mlp_save['rotation_scale'] = rt_s
        mlp_save['rotation_min'] = rt_m
        np.savez_compressed(os.path.join(path, "mlps.npz"), **mlp_save)

        # Calculate and print total size
        size = 0
        size += os.path.getsize(os.path.join(path, "xyz.bin"))
        print("xyz.bin", os.path.getsize(os.path.join(path, "xyz.bin")) / 1e6)
        size += os.path.getsize(os.path.join(path, "f_dc.npz"))
        print("f_dc.npz", os.path.getsize(os.path.join(path, "f_dc.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "scale_base.npz"))
        print("scale_base.npz", os.path.getsize(os.path.join(path, "scale_base.npz")) / 1e6)
        
        # Calculate total size of grid archive (prefer combined file)
        grid_total_size = 0
        grid_all_path = os.path.join(path, "grid_all.npz")
        if os.path.exists(grid_all_path):
            grid_total_size = os.path.getsize(grid_all_path)
            print("grid_all.npz", grid_total_size / 1e6)
        else:
            for filename in os.listdir(path):
                if filename.startswith("grid_") and filename.endswith(".npz"):
                    file_path = os.path.join(path, filename)
                    file_size = os.path.getsize(file_path)
                    grid_total_size += file_size
                    print(f"{filename}", file_size / 1e6)
        size += grid_total_size
        print(f"grid_total (all components)", grid_total_size / 1e6)
        
        size += os.path.getsize(os.path.join(path, "sh_mask.npz"))
        print("sh_mask.npz", os.path.getsize(os.path.join(path, "sh_mask.npz")) / 1e6)
        # mlps are saved in a single file now
        if os.path.exists(os.path.join(path, "mlps.npz")):
            size += os.path.getsize(os.path.join(path, "mlps.npz"))
            print("mlps.npz", os.path.getsize(os.path.join(path, "mlps.npz")) / 1e6)
        else:
            # backward-compatible individual mlp files
            for name in ["opacity.npz", "f_rest.npz", "scale.npz", "rotation.npz"]:
                if os.path.exists(os.path.join(path, name)):
                    size += os.path.getsize(os.path.join(path, name))
                    print(name, os.path.getsize(os.path.join(path, name)) / 1e6)

        print(f"Total size: {(size / 1e6)} MB")

    def decompress_attributes(self, path):
        # Load and verify configuration
        config_path = os.path.join(path, 'config.npz')
        if os.path.exists(config_path):
            config = np.load(config_path)
            # Verify config matches current model (print warnings if mismatch)
            saved_feature_role_split = bool(config['feature_role_split'])
            if saved_feature_role_split != self._feature_role_split:
                print(f"[WARNING] Model feature_role_split mismatch: saved={saved_feature_role_split}, current={self._feature_role_split}")
            # Migration: old configs used encoder_variant; new ones use enable_vm
            if 'enable_vm' in config:
                saved_enable_vm = bool(config['enable_vm'])
                if saved_enable_vm != self._enable_vm:
                    print(f"[WARNING] enable_vm mismatch: saved={saved_enable_vm}, current={self._enable_vm}")
            elif 'encoder_variant' in config:
                old_variant = str(config['encoder_variant'])
                inferred_vm = (old_variant != 'hash_only')
                if inferred_vm != self._enable_vm:
                    print(f"[WARNING] Inferred enable_vm={inferred_vm} from legacy encoder_variant={old_variant}, current={self._enable_vm}")
            # Migration: old configs used mass_aware; new ones use enable_mass_aware
            if 'enable_mass_aware' in config:
                saved_ma = bool(config['enable_mass_aware'])
                if saved_ma != self._enable_mass_aware:
                    print(f"[WARNING] enable_mass_aware mismatch: saved={saved_ma}, current={self._enable_mass_aware}")
            elif 'mass_aware' in config:
                saved_ma = bool(config['mass_aware'])
                if saved_ma != self._enable_mass_aware:
                    print(f"[WARNING] Inferred enable_mass_aware={saved_ma} from legacy mass_aware key, current={self._enable_mass_aware}")
            # Support both old 'glf_*' and new 'geo_*' config key names
            if 'geo_resolution' in config:
                saved_geo_resolution = int(config['geo_resolution'])
            elif 'glf_resolution' in config:
                saved_geo_resolution = int(config['glf_resolution'])
            else:
                saved_geo_resolution = self._geo_resolution
            if saved_geo_resolution != self._geo_resolution:
                print(f"[WARNING] GeoEncoder resolution mismatch: saved={saved_geo_resolution}, current={self._geo_resolution}")
        
        # G-PCC decoding
        xyz = decode_xyz(path)
        self._xyz = torch.from_numpy(np.asarray(xyz)).cuda()

        # Legacy uniform dequantization (kept for backward compatibility)
        if os.path.exists(os.path.join(path, 'f_dc.npz')):
            f_dc_dict = np.load(os.path.join(path, 'f_dc.npz'))
            f_dc = dequantize(f_dc_dict['data'], f_dc_dict['scale'], f_dc_dict['min'])
            self._features_dc = torch.from_numpy(np.asarray(f_dc)).cuda()
        if os.path.exists(os.path.join(path, 'scale_base.npz')):
            scale_base_dict = np.load(os.path.join(path, 'scale_base.npz'))
            scaling_base = dequantize(scale_base_dict['data'], scale_base_dict['scale'], scale_base_dict['min'])
            self._scaling_base = torch.from_numpy(np.asarray(scaling_base)).cuda()
        
        grid_state_dict = {}
        # Prefer loading combined grid archive if present
        grid_all_path = os.path.join(path, "grid_all.npz")
        if os.path.exists(grid_all_path):
            grid_npz = np.load(grid_all_path)
            # find unique normalized keys by scanning files ending with _data
            for name in grid_npz.files:
                if not name.endswith("_data"):
                    continue
                base = name[:-5]  # remove '_data'
                data = grid_npz[f"{base}_data"]
                scale = grid_npz[f"{base}_scale"]
                mn = grid_npz[f"{base}_min"]
                grid_params = dequantize(data, scale, mn, log=True)
                orig_arr = grid_npz[f"{base}_orig"]
                orig_key = orig_arr.item() if np.asarray(orig_arr).shape == () else str(orig_arr)
                grid_state_dict[orig_key] = torch.from_numpy(np.asarray(grid_params)).half().cuda()
        else:
            for filename in os.listdir(path):
                if filename.startswith("grid_") and filename.endswith(".npz"):
                    grid_dict = np.load(os.path.join(path, filename))
                    grid_params = dequantize(grid_dict['data'], grid_dict['scale'], grid_dict['min'], log=True)

                    orig_key = grid_dict['orig_key'].item() if grid_dict['orig_key'].shape == () else str(grid_dict['orig_key'])
                    grid_state_dict[orig_key] = torch.from_numpy(np.asarray(grid_params)).half().cuda()

        if grid_state_dict:
            # Lenient load to stay compatible with older checkpoints
            target_sd = self._grid.state_dict()
            filtered_sd = {}
            skipped_keys = []
            mismatched = []

            for key, tensor in grid_state_dict.items():
                if key not in target_sd:
                    skipped_keys.append(key)
                    continue
                if target_sd[key].shape != tensor.shape:
                    mismatched.append((key, tuple(tensor.shape), tuple(target_sd[key].shape)))
                    continue
                filtered_sd[key] = tensor

            if skipped_keys:
                print(f"[WARN] Ignoring unexpected grid params: {skipped_keys}")
            if mismatched:
                for name, saved_shape, model_shape in mismatched:
                    print(f"[WARN] Grid param shape mismatch for {name}: saved {saved_shape} vs model {model_shape}; skipping")
            missing_after_filter = [k for k in target_sd.keys() if k not in filtered_sd]
            if missing_after_filter:
                print(f"[WARN] Missing grid params; using model defaults for: {missing_after_filter}")

            self._grid.load_state_dict(filtered_sd, strict=False)

        # sh_mask
        sh_mask = np.load(os.path.join(path, 'sh_mask.npz'))['data'].astype(np.float32)
        self._sh_mask = torch.from_numpy(np.asarray(sh_mask)).cuda()
        
        # opacity mask
        mask_path = os.path.join(path, 'mask.npz')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)['data'].astype(np.float32)
            self._mask = torch.from_numpy(np.asarray(mask)).cuda()
        else:
            # Legacy checkpoint: default to all-ones mask
            N = self._xyz.shape[0]
            self._mask = torch.ones((N, 1), device="cuda")

        # -------- Quality-first load (preferred) --------
        mlps_fp16_path = os.path.join(path, 'mlps_fp16.npz')
        if os.path.exists(mlps_fp16_path):
            mlp_npz = _load_npz(mlps_fp16_path)
            _strict_assign_param(self._opacity_head, _np_to_torch(mlp_npz['opacity'], device='cuda', dtype=torch.float16), "opacity_head.params", mlps_fp16_path)
            _strict_assign_param(self._scaling_head, _np_to_torch(mlp_npz['scaling'], device='cuda', dtype=torch.float16), "scaling_head.params", mlps_fp16_path)
            _strict_assign_param(self._rotation_head, _np_to_torch(mlp_npz['rotation'], device='cuda', dtype=torch.float16), "rotation_head.params", mlps_fp16_path)
            _strict_assign_param(self._features_rest_head, _np_to_torch(mlp_npz['features_rest'], device='cuda', dtype=torch.float16), "features_rest_head.params", mlps_fp16_path)
        else:
            # -------- Legacy quantized load (compatibility) --------
            mlps_path = os.path.join(path, 'mlps.npz')
            def _coerce_and_assign(head, np_array, name):
                print(f"[WARN] Legacy quantized '{name}' may reduce quality; consider regenerating compression.")
                flat = np.asarray(np_array).ravel()
                expected = head.state_dict()['params'].numel()
                if flat.size != expected:
                    print(f"[WARN] Loaded '{name}' params size ({flat.size}) != expected ({expected}), coercing")
                    if flat.size > expected:
                        flat = flat[:expected]
                    else:
                        pad = np.zeros(expected - flat.size, dtype=flat.dtype)
                        flat = np.concatenate([flat, pad])
                tensor = torch.from_numpy(flat).half().cuda()
                head.params = nn.Parameter(tensor)

            if os.path.exists(mlps_path):
                mlp_npz = np.load(mlps_path)
                o = dequantize(mlp_npz['opacity_data'], mlp_npz['opacity_scale'], mlp_npz['opacity_min'], log=True)
                fr = dequantize(mlp_npz['f_rest_data'], mlp_npz['f_rest_scale'], mlp_npz['f_rest_min'], log=True)
                sc = dequantize(mlp_npz['scale_data'], mlp_npz['scale_scale'], mlp_npz['scale_min'], log=True)
                rt = dequantize(mlp_npz['rotation_data'], mlp_npz['rotation_scale'], mlp_npz['rotation_min'], log=True)
                _coerce_and_assign(self._opacity_head, o, 'opacity')
                _coerce_and_assign(self._features_rest_head, fr, 'features_rest')
                _coerce_and_assign(self._scaling_head, sc, 'scaling')
                _coerce_and_assign(self._rotation_head, rt, 'rotation')
            else:
                # backward compatible loading of individual float files
                opacity_params = np.load(os.path.join(path, 'opacity.npz'))['data']
                f_rest_params = np.load(os.path.join(path, 'f_rest.npz'))['data']
                scale_params = np.load(os.path.join(path, 'scale.npz'))['data']
                rotation_params = np.load(os.path.join(path, 'rotation.npz'))['data']
                _coerce_and_assign(self._opacity_head, opacity_params, 'opacity')
                _coerce_and_assign(self._features_rest_head, f_rest_params, 'features_rest')
                _coerce_and_assign(self._scaling_head, scale_params, 'scaling')
                _coerce_and_assign(self._rotation_head, rotation_params, 'rotation')

        # Quality-first overrides for base attributes if present
        f_dc_fp16_path = os.path.join(path, 'f_dc_fp16.npz')
        if os.path.exists(f_dc_fp16_path):
            f_dc_npz = _load_npz(f_dc_fp16_path)
            _strict_assign(self._features_dc, _np_to_torch(f_dc_npz['data'], device='cuda', dtype=torch.float16), "features_dc", f_dc_fp16_path)
        scale_base_fp16_path = os.path.join(path, 'scale_base_fp16.npz')
        if os.path.exists(scale_base_fp16_path):
            sb_npz = _load_npz(scale_base_fp16_path)
            _strict_assign(self._scaling_base, _np_to_torch(sb_npz['data'], device='cuda', dtype=torch.float16), "scaling_base", scale_base_fp16_path)
        mask_logits_fp16_path = os.path.join(path, 'mask_logits_fp16.npz')
        if os.path.exists(mask_logits_fp16_path):
            m_npz = _load_npz(mask_logits_fp16_path)
            _strict_assign(self._mask, _np_to_torch(m_npz['data'], device='cuda', dtype=torch.float16), "mask", mask_logits_fp16_path)
        sh_mask_logits_fp16_path = os.path.join(path, 'sh_mask_logits_fp16.npz')
        if os.path.exists(sh_mask_logits_fp16_path):
            sm_npz = _load_npz(sh_mask_logits_fp16_path)
            _strict_assign(self._sh_mask, _np_to_torch(sm_npz['data'], device='cuda', dtype=torch.float16), "sh_mask", sh_mask_logits_fp16_path)
        vm_planes_fp16_path = os.path.join(path, 'vm_planes_fp16.npz')
        if os.path.exists(vm_planes_fp16_path) and hasattr(self._grid, 'plane_xy'):
            # TODO(stage1-task4): ensure compression/export pipeline supports hash_only without vm_planes artifacts
            vm_npz = _load_npz(vm_planes_fp16_path)
            _strict_assign(self._grid.plane_xy, _np_to_torch(vm_npz['plane_xy'], device='cuda', dtype=torch.float16), "plane_xy", vm_planes_fp16_path)
            _strict_assign(self._grid.plane_xz, _np_to_torch(vm_npz['plane_xz'], device='cuda', dtype=torch.float16), "plane_xz", vm_planes_fp16_path)
            _strict_assign(self._grid.plane_yz, _np_to_torch(vm_npz['plane_yz'], device='cuda', dtype=torch.float16), "plane_yz", vm_planes_fp16_path)

        self.active_sh_degree = self.max_sh_degree

    def load_attributes(self, path):
        self.decompress_attributes(path)
        self.update_attributes()
        sh_mask = self._sh_mask
        sh_mask_degree = torch.ones_like(self._features_rest) # (N, 15, 3)
        for deg in range(1, self.active_sh_degree + 1):
            sh_mask_degree[:, deg**2 - 1:, :] *= sh_mask[:, deg - 1:deg].unsqueeze(1)
        
        self._features_rest *= sh_mask_degree
        self.sh_levels = sh_mask.sum(1).int()
