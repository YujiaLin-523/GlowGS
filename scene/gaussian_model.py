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
        
        # Create encoder using factory for ablation study support
        # encoder_variant is set via __init__ from training args
        self._grid = create_gaussian_encoder(
            variant=self._encoder_variant,
            encoding_config=self.encoding_config,
            network_config=self.network_config,
            geo_resolution=self._geo_resolution,
            geo_rank=self._geo_rank,
            geo_channels=self._geo_channels,
            feature_mod_type=self._feature_mod_type,
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
        base_dim, geometry_dim, appearance_dim = get_encoder_output_dims(self._grid, self._encoder_variant)
        
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
                 geo_resolution=48, geo_rank=6, geo_channels=8, encoder_variant="hybrid",
                 densify_strategy="feature_weighted", feature_mod_type="film",
                 mass_aware_scale: float = 0.1):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # Store encoder variant and densification strategy for ablation studies
        self._encoder_variant = encoder_variant
        self._densify_strategy = densify_strategy
        self._feature_role_split = feature_role_split  # Geometry/appearance disentanglement flag
        self._feature_mod_type = feature_mod_type  # 'film' or 'concat' for ECCV ablation
        # Mass-aware gradient weighting scale (xi): higher = stronger pruning, lower = softer
        self._mass_aware_scale = mass_aware_scale
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
        # QAT config
        self.qat_enabled = False
        self.qat_bits = {
            "grid": 6,
            "mlp": 8,
            "dc": 8,
            "scale": 8,
            "rot": 8,
            "opacity": 8,
            "sh": 8,
        }
        self.qat_mask_th = 0.01
        self.qat_log_grid = True
        # Precomputed attribute cache for fast inference (populated by precompute_attributes)
        self._precomputed_cache = None
        self.setup_functions()
        self.setup_configs(hash_size, width, depth)
        self.setup_params()


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

    # === QAT helpers ===
    def enable_qat(self, bit_grid=6, bit_mlp=8, bit_dc=8, bit_scale=8, bit_rot=8, bit_opacity=8, bit_sh=8, mask_th=0.01, log_grid=True):
        self.qat_enabled = True
        self.qat_bits.update({
            "grid": bit_grid,
            "mlp": bit_mlp,
            "dc": bit_dc,
            "scale": bit_scale,
            "rot": bit_rot,
            "opacity": bit_opacity,
            "sh": bit_sh,
        })
        self.qat_mask_th = mask_th
        self.qat_log_grid = log_grid

    def _fake_quant(self, x: torch.Tensor, bit: int = 8, log: bool = False, eps: float = 1e-8):
        if bit <= 0:
            return x
        with torch.no_grad():
            x_det = x.detach()
            if log:
                x_sign = torch.sign(x_det)
                x_log = torch.log1p(torch.abs(x_det))
                xmin = x_log.min()
                xmax = x_log.max()
                scale = (xmax - xmin + eps) / (2 ** bit - 1)
                x_q = torch.clamp(torch.round((x_log - xmin) / scale), 0, 2 ** bit - 1)
                x_hat = torch.expm1(x_q * scale + xmin) * x_sign
            else:
                xmin = x_det.min()
                xmax = x_det.max()
                scale = (xmax - xmin + eps) / (2 ** bit - 1)
                x_q = torch.clamp(torch.round((x_det - xmin) / scale), 0, 2 ** bit - 1)
                x_hat = x_q * scale + xmin
        # STE: keep gradient of original x
        return x + (x_hat - x).detach()
    
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

    def update_attributes(self, force_update=False):
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
            # cached values already quantized (if QAT enabled during precompute)
            return
        
        # 3DGS mode: skip encoder, use explicit SH parameters only
        # In this mode, _features_rest is directly optimized (no encoder)
        if self._encoder_variant == '3dgs':
            # Initialize implicit attributes to defaults if not already set
            N = self._xyz.shape[0]
            if N == 0:
                return
            
            # Set default geometry attributes (optimized separately)
            if self._scaling.shape[0] != N:
                self._scaling = torch.zeros((N, 3), device="cuda")
            if self._rotation.shape[0] != N:
                self._rotation = self._rotation_init.expand(N, -1).clone()
            if self._opacity.shape[0] != N:
                self._opacity = self._opacity_init.expand(N, -1).clone()
            
            # _features_rest is directly optimized in 3DGS mode (no encoder)
            # Just ensure it exists and has the right shape
            if self.max_sh_degree > 0 and self._features_rest.shape[0] != N:
                sh_dim = (self.max_sh_degree + 1) ** 2 - 1
                self._features_rest = torch.zeros((N, sh_dim, 3), device="cuda")
            
            return  # Early exit for 3DGS mode
        
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
        
        contracted_xyz = self.get_contracted_xyz

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
                coords = contracted_xyz[i: end_idx]
                encoder_out = self._grid(coords)

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
                    if self.qat_enabled:
                        fused = self._fake_quant(fused, bit=self.qat_bits["grid"], log=self.qat_log_grid)
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
                if self.qat_enabled:
                    geometry_chunk = self._fake_quant(geometry_chunk, bit=self.qat_bits["grid"], log=self.qat_log_grid)
                    appearance_chunk = self._fake_quant(appearance_chunk, bit=self.qat_bits["grid"], log=self.qat_log_grid)
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
            if self.max_sh_degree > 0:
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
                coords = contracted_xyz[i: end_idx]
                encoder_out = self._grid(coords)
                feats = encoder_out if not isinstance(encoder_out, tuple) else encoder_out[0]
                if self.qat_enabled:
                    feats = self._fake_quant(feats, bit=self.qat_bits["grid"], log=self.qat_log_grid)
                # Encoder already clamps, skip redundant clamp
                opacity_chunks.append(self._opacity_head(feats) + self._opacity_init)
                scaling_chunks.append(self._scaling_head(feats))
                rotation_chunks.append(self._rotation_head(feats) + self._rotation_init)
                if self.max_sh_degree > 0:
                    features_rest_chunks.append(self._features_rest_head(feats).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3))

            self._opacity = torch.cat(opacity_chunks, dim=0)
            self._scaling = torch.cat(scaling_chunks, dim=0)
            self._rotation = torch.cat(rotation_chunks, dim=0)
            if self.max_sh_degree > 0:
                self._features_rest = torch.cat(features_rest_chunks, dim=0)
            
            # Update cache hash after successful computation
            self._cached_attributes_hash = (self._xyz.data_ptr(), N, self._xyz.requires_grad)
        
        # Apply QAT fake quant + STE masks once per update
        if self.qat_enabled:
            self._opacity = self._fake_quant(self._opacity, bit=self.qat_bits["opacity"])
            self._scaling = self._fake_quant(self._scaling, bit=self.qat_bits["scale"])
            self._rotation = torch.nn.functional.normalize(
                self._fake_quant(self._rotation, bit=self.qat_bits["rot"]), dim=-1
            )
            if self.max_sh_degree > 0:
                self._features_rest = self._fake_quant(self._features_rest, bit=self.qat_bits["sh"])
            # Explicit params
            self._features_dc = self._fake_quant(self._features_dc, bit=self.qat_bits["dc"])

            # Masks: single threshold + STE (skip if preactivated from compressed ckpt)
            if not getattr(self, '_masks_preactivated', False):
                mask_soft = self.mask_activation(self._mask)
                mask_hard = (mask_soft > self.qat_mask_th).float()
                self._mask_q = mask_hard + (mask_soft - mask_soft.detach())
                sh_mask_soft = self.sh_mask_activation(self._sh_mask)
                sh_mask_hard = (sh_mask_soft > self.qat_mask_th).float()
                self._sh_mask_q = sh_mask_hard + (sh_mask_soft - sh_mask_soft.detach())

        # NOTE: Removed torch.cuda.empty_cache() here for real-time performance
        # empty_cache() is expensive (~1-2ms) and should only be called when necessary
    
    def precompute_attributes(self):
        """
        Precompute and cache all Gaussian attributes for fast inference.
        
        Call this once after model loading to avoid per-frame MLP inference.
        The cached attributes will be used by update_attributes() automatically.
        """
        if self._encoder_variant == '3dgs':
            print("[Precompute] 3DGS mode: no encoder decoding needed")
            return
        
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
        print(f"[Precompute] Cached {n_points:,} Gaussian attributes for fast rendering")
    
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
        base = self.scaling_base_activation(clamped_scaling)
        if self.qat_enabled:
            base = self._fake_quant(base, bit=self.qat_bits["scale"])
        return base

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
        rot = self.rotation_activation(rotation)
        if self.qat_enabled:
            rot = torch.nn.functional.normalize(self._fake_quant(rot, bit=self.qat_bits["rot"]), dim=-1)
        return rot
    
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
            feats = torch.cat((features_dc, features_rest), dim=1)
        else:
            feats = features_dc
        if self.qat_enabled and feats.numel() > 0:
            feats = self._fake_quant(feats, bit=self.qat_bits["mlp"])
        return feats
    
    @property
    def get_opacity(self):
        op = self.opacity_activation(self._opacity)
        if self.qat_enabled:
            op = self._fake_quant(op, bit=self.qat_bits["opacity"])
        return op
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_mask(self):
        # Skip sigmoid if masks were loaded from compressed checkpoint (already 0/1)
        if getattr(self, '_masks_preactivated', False):
            return self._mask
        if self.qat_enabled and hasattr(self, "_mask_q"):
            return self._mask_q
        return self.mask_activation(self._mask)

    @property
    def get_sh_mask(self):
        # Skip sigmoid if masks were loaded from compressed checkpoint (already 0/1)
        if getattr(self, '_masks_preactivated', False):
            return self._sh_mask
        if self.qat_enabled and hasattr(self, "_sh_mask_q"):
            return self._sh_mask_q
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

        # In 3DGS mode, initialize _features_rest as explicit parameters
        if self._encoder_variant == '3dgs' and self.max_sh_degree > 0:
            # Initialize higher-order SH coefficients to zero
            self._features_rest = nn.Parameter(
                features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
            )
        else:
            # In encoder-based modes, _features_rest is implicit (generated by encoder)
            # Initialize to empty tensor (will be populated by update_attributes)
            self._features_rest = torch.empty(0)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
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
        
        # In 3DGS mode, add _features_rest as explicit optimizable parameter
        if self._encoder_variant == '3dgs' and isinstance(self._features_rest, nn.Parameter):
            param_groups.append({
                'params': [self._features_rest], 
                'lr': training_args.feature_rest_lr_init, 
                "name": "f_rest"
            })
        
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
                # In 3DGS mode, f_rest is in explicit optimizer
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
            param_tensor = group["params"][0]
            # If shapes mismatch (can happen after densify/prune races), align mask length
            if param_tensor.shape[0] != mask.shape[0]:
                # truncate or pad mask to param length to avoid crash; logs kept minimal
                if param_tensor.shape[0] < mask.shape[0]:
                    local_mask = mask[:param_tensor.shape[0]]
                else:
                    pad = torch.zeros((param_tensor.shape[0] - mask.shape[0],), device=mask.device, dtype=mask.dtype).bool()
                    local_mask = torch.cat([mask, pad], dim=0)
            else:
                local_mask = mask
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][local_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][local_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][local_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][local_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Prune Gaussians based on boolean mask.
        
        Args:
            mask: Boolean tensor where True indicates points to REMOVE
        """
        # Ensure mask is on CUDA for consistency
        if not mask.is_cuda:
            mask = mask.cuda()
        # Align mask length to current xyz length to avoid shape mismatches
        n_xyz = self._xyz.shape[0]
        if mask.shape[0] != n_xyz:
            if mask.shape[0] > n_xyz:
                mask = mask[:n_xyz]
            else:
                pad = torch.zeros((n_xyz - mask.shape[0],), device=mask.device, dtype=mask.dtype).bool()
                mask = torch.cat([mask, pad], dim=0)
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
            
            # Prune f_rest if applicable
            if self._encoder_variant == '3dgs':
                if isinstance(self._features_rest, (torch.Tensor, nn.Parameter)) and self._features_rest.numel() > 0:
                    self._features_rest = safe_prune(self._features_rest, valid_points_mask, "f_rest")
            else:
                # For encoder models, prune if it exists and matches shape
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
            
            # In 3DGS mode, also prune f_rest
            if self._encoder_variant == '3dgs' and "f_rest" in optimizable_tensors:
                self._features_rest = optimizable_tensors["f_rest"]

            # Training buffers - these MUST exist and match in training mode
            def _safe_slice(buf):
                if buf is None or not isinstance(buf, torch.Tensor) or buf.numel() == 0:
                    return buf
                if buf.shape[0] != valid_points_mask.shape[0]:
                    # align mask again to buffer length
                    vm = valid_points_mask
                    if vm.shape[0] > buf.shape[0]:
                        vm = vm[:buf.shape[0]]
                    else:
                        pad = torch.zeros((buf.shape[0] - vm.shape[0],), device=vm.device, dtype=vm.dtype).bool()
                        vm = torch.cat([vm, pad], dim=0)
                else:
                    vm = valid_points_mask
                return buf[vm]

            self.xyz_gradient_accum = _safe_slice(self.xyz_gradient_accum)
            self.denom = _safe_slice(self.denom)
            self.max_radii2D = _safe_slice(self.max_radii2D)
        
        # Clear cached contracted_xyz since xyz changed
        if hasattr(self, '_cached_contracted_xyz'):
            delattr(self, '_cached_contracted_xyz')

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

    def densification_postfix(self, new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest=None):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "scaling_base": new_scaling_base,
            "mask": new_mask,
            "sh_mask": new_sh_mask,
        }
        
        # In 3DGS mode, explicitly add f_rest to optimizer
        if self._encoder_variant == '3dgs' and new_features_rest is not None:
            d["f_rest"] = new_features_rest

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling_base = optimizable_tensors["scaling_base"]
        self._mask = optimizable_tensors["mask"]
        self._sh_mask = optimizable_tensors["sh_mask"]
        
        # In 3DGS mode, update _features_rest from optimizer
        if self._encoder_variant == '3dgs' and "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # Invalidate attribute caches; shapes changed
        self._cached_attributes_hash = None
        if hasattr(self, '_cached_contracted_key'):
            self._cached_contracted_key = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        Split large Gaussians with high gradient.
        Uses original 3DGS logic with FPS guard for screen-space size.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected_pts_mask = padded_grad >= grad_threshold
        
        # World-space size constraint (original 3DGS)
        is_large_ws = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        # Screen-space size constraint (FPS optimization)
        is_large_ss = self.max_radii2D > 3.0
        
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_large_ws)
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_large_ss)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_scaling_base = self.scaling_base_inverse_activation(self.get_scaling_base[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_mask = self._mask[selected_pts_mask].repeat(N,1)
        new_sh_mask = self._sh_mask[selected_pts_mask].repeat(N,1)
        
        new_features_rest = None
        if self._encoder_variant == '3dgs' and isinstance(self._features_rest, nn.Parameter):
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Clone small Gaussians with high gradient.
        Uses original 3DGS logic with FPS guard for screen-space size.
        """
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # World-space size constraint (original 3DGS)
        is_small_ws = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        # Screen-space size constraint (FPS optimization)
        is_visible_ss = self.max_radii2D > 2.0
        
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_small_ws)
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_visible_ss)
        selected_pts_mask = torch.logical_and(selected_pts_mask, is_visible_ss)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_scaling_base = self._scaling_base[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_sh_mask = self._sh_mask[selected_pts_mask]
        
        new_features_rest = None
        if self._encoder_variant == '3dgs' and isinstance(self._features_rest, nn.Parameter):
            new_features_rest = self._features_rest[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest)
        self.update_attributes(force_update=True)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration=0):
        """
        Core densification and pruning logic.
        Simplified to use original 3DGS approach with opacity-based pruning.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.update_attributes(force_update=True)
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        self.update_attributes(force_update=True)
        
        # === Pruning: Size-Aware Opacity Thresholds with Ramp ===
        opacity = self.get_opacity.squeeze()
        
        # Base pruning: extremely low opacity (original 3DGS)
        prune_mask = (opacity < min_opacity)
        
        # FIX: Size-aware pruning with smooth ramp (4k-6k iterations)
        # Use continuous threshold transition to avoid step discontinuity
        size_prune_start = getattr(self, 'size_prune_start_iter', 4000)
        size_prune_end = getattr(self, 'size_prune_end_iter', 6000)
        
        # Ramp weight: 0 at start, 1 at end (linear transition)
        ramp_w = max(0.0, min(1.0, (iteration - size_prune_start) / max(1, size_prune_end - size_prune_start)))
        
        # Smooth radius threshold transition: 1e9 (never trigger) -> 50.0 (original threshold)
        # This ensures w=0 behaves as "disabled", w=1 as "fully enabled"
        radius_threshold = (1.0 - ramp_w) * 1e9 + ramp_w * 50.0
        
        # Size-Aware Pruning: penalize large semi-transparent volumes (now always applied with smooth threshold)
        # - Small radius (<50px): details/spokes/background, keep even if semi-transparent
        # - Large radius (>50px): must be solid (>0.005) or get pruned
        # TUNED: Relaxed from (20px, 0.05) to (50px, 0.005) to preserve background/sky
        large_radius_mask = self.max_radii2D > radius_threshold
        semi_transparent_mask = opacity < 0.005
        volumetric_bloat = torch.logical_and(large_radius_mask, semi_transparent_mask)
        prune_mask = torch.logical_or(prune_mask, volumetric_bloat)
        
        # Additional artifact cleanup for extremely large points
        if max_screen_size:
            is_huge_screen = self.max_radii2D > max_screen_size
            is_transparent = opacity < 0.5
            is_artifact = torch.logical_and(is_huge_screen, is_transparent)
            prune_mask = torch.logical_or(prune_mask, is_artifact)
        
        self.prune_points(prune_mask)
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
            self.densify_and_prune(adaptive_grad, min_opacity, extent, max_screen_size, iteration=iteration)
            new_count = self.get_xyz.shape[0]
            # Log capacity warning if approaching limit
            if new_count >= max_points * 0.9:
                print(f"[CAPACITY] Approaching limit: {new_count}/{max_points} ({new_count/max_points*100:.1f}%)")
        elif iteration >= densify_until:
            # Post-densification phase: periodic mask pruning only
            if iteration % prune_interval == 0:
                self.mask_prune(mask_threshold=mask_prune_threshold)
        else:
            # At or over capacity: skip densification, still allow pruning
            if iteration % prune_interval == 0:
                self.mask_prune(mask_threshold=mask_prune_threshold)
                print(f"[CAPACITY] At limit ({num_points}>={max_points}), densify skipped, prune only")

    def mask_prune(self, mask_threshold=0.01):
        # Pruning: points with mask <= threshold are removed (LocoGS: 0.01)
        prune_mask = (self.get_mask <= mask_threshold).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opacity=None, radii=None, iteration=0):
        """
        Accumulate gradients for densification decision using Mass-Aware Gradient Weighting.
        
        Theory (for paper):
        We decompose gradient confidence into two orthogonal factors:
        1. Visibility Boost (sqrt(alpha)): Non-linear rescaling to compensate for 
           sub-pixel structures that appear semi-transparent due to rasterization averaging.
           This prevents fine details (bike spokes) from being ignored.
        2. Mass Regularization (1/(1 + alpha*r/tau)): Penalizes large, solid regions 
           where high gradients often indicate lighting variation rather than geometric 
           deficiency. This prevents over-densification on smooth surfaces (roads).
        
        Formula: weight = sqrt(alpha) / (1 + alpha * radius / tau)
        
        Args:
            viewspace_point_tensor: Tensor with .grad containing screen-space gradients
            update_filter: Boolean mask for visible points
            opacity: Optional opacity tensor [N, 1]. If provided, used for weighting.
            radii: Optional screen-space radii tensor [N]. If provided, used for mass regularization.
            iteration: Current training iteration (for warmup schedule)
        """
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        
        # Only apply Mass-Aware weighting when densify_strategy is 'feature_weighted'
        use_mass_aware = (self._densify_strategy == 'feature_weighted')
        
        # FIX: Mass-aware with smooth ramp (3k-5k iterations)
        # Prevents abrupt gradient distribution change at 5k iteration
        mass_start = getattr(self, 'mass_aware_start_iter', 3000)
        mass_end = getattr(self, 'mass_aware_end_iter', 5000)
        
        # Compute ramp weight: 0->1 over [start, end]
        ramp_w = self._compute_ramp_weight(iteration, mass_start, mass_end)
        
        if use_mass_aware and opacity is not None and radii is not None:
            # Mass-Aware Gradient Weighting (GlowGS innovation)
            alpha = opacity[update_filter].clamp(0.01, 1.0)  # [N_visible, 1]
            r = radii[update_filter].float().unsqueeze(-1).clamp(min=1.0)  # [N_visible, 1]
            
            tau = 100.0
            
            # Visibility boost: sqrt(alpha) - helps small/thin structures with low opacity
            visibility_boost = torch.sqrt(alpha)
            
            # Mass penalty: computed but smoothly ramped in
            mass = alpha * r
            mass_penalty = 1.0 / (1.0 + self._mass_aware_scale * (mass / tau))
            
            # Smooth interpolation: w=0 -> weight=visibility_boost (no penalty)
            #                       w=1 -> weight=visibility_boost*mass_penalty (full penalty)
            # Equivalent to: weight = visibility_boost * (1 + w * (mass_penalty - 1))
            weight = visibility_boost * (1.0 + ramp_w * (mass_penalty - 1.0))
            
            grad_norm = grad_norm * weight
            
        elif opacity is not None and use_mass_aware:
            # Fallback: simple opacity weighting (backward compatible)
            opacity_weight = opacity[update_filter].clamp(0.0, 1.0)
            grad_norm = grad_norm * opacity_weight
        # else: Standard 3DGS densification - no weighting applied
        
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
        }
        np.savez(os.path.join(path, "config.npz"), **config)
        
        # Save the complete GeometryAppearanceEncoder state_dict
        # This includes hash_encoder, geo_encoder, and fusion_layer parameters
        # All files use underscore naming format for consistency
        
        grid_state_dict = self._grid.state_dict()
        grid_save_dict = {}
        grid_bit = self.qat_bits.get("grid", 6)
        grid_log = self.qat_log_grid
        for key, value in grid_state_dict.items():
            normalized_key = key.replace('.', '_')
            param_data = value.detach().cpu().numpy()
            param_quant, param_scale, param_min = quantize(param_data, bit=grid_bit, log=grid_log)
            grid_save_dict[f"{normalized_key}_data"] = param_quant.astype(np.uint8)
            grid_save_dict[f"{normalized_key}_scale"] = param_scale
            grid_save_dict[f"{normalized_key}_min"] = param_min
            grid_save_dict[f"{normalized_key}_orig"] = np.array(key)
        np.savez_compressed(os.path.join(path, "grid_all.npz"), **grid_save_dict)

        # G-PCC encoding
        encode_xyz(xyz, path)

        # uniform quantization (bits from QAT config)
        f_dc_quant, f_dc_scale, f_dc_min = quantize(f_dc, bit=self.qat_bits.get("dc", 8))
        scaling_base_quant, scaling_base_scale, scaling_base_min = quantize(scale_base, bit=self.qat_bits.get("scale", 8))

        np.savez_compressed(os.path.join(path, "f_dc.npz"), data=f_dc_quant.astype(np.uint8), scale=f_dc_scale, min=f_dc_min)
        np.savez_compressed(os.path.join(path, "scale_base.npz"), data=scaling_base_quant.astype(np.uint8), scale=scaling_base_scale, min=scaling_base_min)
        
        # sh mask (single threshold)
        sh_mask = (sh_mask > self.qat_mask_th).cumprod(axis=-1).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "sh_mask.npz"), data=sh_mask)
        
        # opacity mask (binarized to save space)
        mask = self.get_mask.detach().cpu().numpy()
        mask_bin = (mask > self.qat_mask_th).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "mask.npz"), data=mask_bin)

        # mlps: quantize MLP parameter tensors (configurable bit)
        mlp_save = {}
        opacity_params = self._opacity_head.state_dict()['params'].cpu().detach().numpy()
        f_rest_params = self._features_rest_head.state_dict()['params'].cpu().detach().numpy()
        scale_params = self._scaling_head.state_dict()['params'].cpu().detach().numpy()
        rotation_params = self._rotation_head.state_dict()['params'].cpu().detach().numpy()

        bit_mlp = self.qat_bits.get("mlp", 8)
        o_q, o_s, o_m = quantize(opacity_params, bit=bit_mlp, log=False)
        fr_q, fr_s, fr_m = quantize(f_rest_params, bit=bit_mlp, log=False)
        sc_q, sc_s, sc_m = quantize(scale_params, bit=bit_mlp, log=False)
        rt_q, rt_s, rt_m = quantize(rotation_params, bit=bit_mlp, log=False)

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
        for root, _, files in os.walk(path):
            for fn in files:
                size += os.path.getsize(os.path.join(root, fn))
        size_mb = size / 1e6
        print(f"[Compression] Total size (all files): {size_mb:.2f} MB")

    def decompress_attributes(self, path):
        # Load and verify configuration
        config_path = os.path.join(path, 'config.npz')
        if os.path.exists(config_path):
            config = np.load(config_path)
            # Verify config matches current model (print warnings if mismatch)
            saved_feature_role_split = bool(config['feature_role_split'])
            if saved_feature_role_split != self._feature_role_split:
                print(f"[WARNING] Model feature_role_split mismatch: saved={saved_feature_role_split}, current={self._feature_role_split}")
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

        # uniform dequantization
        f_dc_dict = np.load(os.path.join(path, 'f_dc.npz'))
        scale_base_dict = np.load(os.path.join(path, 'scale_base.npz'))

        f_dc = dequantize(f_dc_dict['data'], f_dc_dict['scale'], f_dc_dict['min'])
        scaling_base = dequantize(scale_base_dict['data'], scale_base_dict['scale'], scale_base_dict['min'])

        self._features_dc = torch.from_numpy(np.asarray(f_dc)).cuda()
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

        # sh_mask: compressed file stores post-sigmoid binary 0/1 values
        # _masks_preactivated flag will skip sigmoid in get_sh_mask/get_mask
        sh_mask = np.load(os.path.join(path, 'sh_mask.npz'))['data'].astype(np.float32)
        self._sh_mask = torch.from_numpy(sh_mask).float().cuda()
        
        # opacity mask: same - already post-sigmoid 0/1 values
        mask_path = os.path.join(path, 'mask.npz')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)['data'].astype(np.float32)
            self._mask = torch.from_numpy(mask).float().cuda()
        else:
            # Legacy checkpoint: default to all-ones mask
            N = self._xyz.shape[0]
            self._mask = torch.ones((N, 1), device="cuda", dtype=torch.float32)

        # mlps: prefer combined mlps.npz (quantized) else fallback to older float16 files
        mlps_path = os.path.join(path, 'mlps.npz')
        def _coerce_and_assign(head, np_array, name):
            """Coerce loaded numpy params to the target size expected by the head and assign safely."""
            flat = np.asarray(np_array).ravel()
            try:
                expected = head.state_dict()['params'].numel()
            except Exception:
                # Fallback: try length of params attribute if present
                try:
                    expected = int(head.params.numel())
                except Exception:
                    expected = flat.size

            if flat.size != expected:
                print(f"[WARN] Loaded '{name}' params size ({flat.size}) != expected ({expected}), coercing")
                if flat.size > expected:
                    flat = flat[:expected]
                else:
                    # pad with zeros
                    pad = np.zeros(expected - flat.size, dtype=flat.dtype)
                    flat = np.concatenate([flat, pad])

            tensor = torch.from_numpy(flat).half().cuda()
            # Assign to head in a way compatible with tinycudann
            try:
                head.params = nn.Parameter(tensor)
            except Exception:
                # Last resort: try to set via state_dict
                sd = head.state_dict()
                sd['params'] = tensor
                head.load_state_dict(sd)

        if os.path.exists(mlps_path):
            mlp_npz = np.load(mlps_path)
            o = dequantize(mlp_npz['opacity_data'], mlp_npz['opacity_scale'], mlp_npz['opacity_min'], log=False)
            fr = dequantize(mlp_npz['f_rest_data'], mlp_npz['f_rest_scale'], mlp_npz['f_rest_min'], log=False)
            sc = dequantize(mlp_npz['scale_data'], mlp_npz['scale_scale'], mlp_npz['scale_min'], log=False)
            rt = dequantize(mlp_npz['rotation_data'], mlp_npz['rotation_scale'], mlp_npz['rotation_min'], log=False)

            _coerce_and_assign(self._opacity_head, o, 'opacity')
            _coerce_and_assign(self._features_rest_head, fr, 'features_rest')
            _coerce_and_assign(self._scaling_head, sc, 'scaling')
            _coerce_and_assign(self._rotation_head, rt, 'rotation')
        else:
            # backward compatible loading
            opacity_params = np.load(os.path.join(path, 'opacity.npz'))['data']
            f_rest_params = np.load(os.path.join(path, 'f_rest.npz'))['data']
            scale_params = np.load(os.path.join(path, 'scale.npz'))['data']
            rotation_params = np.load(os.path.join(path, 'rotation.npz'))['data']

            _coerce_and_assign(self._opacity_head, opacity_params, 'opacity')
            _coerce_and_assign(self._features_rest_head, f_rest_params, 'features_rest')
            _coerce_and_assign(self._scaling_head, scale_params, 'scaling')
            _coerce_and_assign(self._rotation_head, rotation_params, 'rotation')

        self.active_sh_degree = self.max_sh_degree

    def load_attributes(self, path):
        self.decompress_attributes(path)
        # Mark masks as pre-activated (already 0/1, skip sigmoid in get_mask/get_sh_mask)
        self._masks_preactivated = True
        # Align mask and sh_mask length with xyz to avoid shape mismatch at render time
        def _align_mask(tensor, target_len, fill=1.0):
            if tensor.shape[0] > target_len:
                return tensor[:target_len]
            if tensor.shape[0] < target_len:
                pad_shape = (target_len - tensor.shape[0],) + tensor.shape[1:]
                pad = torch.full(pad_shape, fill, device=tensor.device, dtype=tensor.dtype)
                return torch.cat([tensor, pad], dim=0)
            return tensor
        target_len = self._xyz.shape[0]
        self._mask = _align_mask(self._mask, target_len, fill=1.0)
        self._sh_mask = _align_mask(self._sh_mask, target_len, fill=1.0)
        self.update_attributes()
        sh_mask = self._sh_mask
        self.sh_levels = (sh_mask > 0.01).float().sum(1).int()
