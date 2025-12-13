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
                 densify_strategy="feature_weighted"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # Store encoder variant and densification strategy for ablation studies
        self._encoder_variant = encoder_variant
        self._densify_strategy = densify_strategy
        self._feature_role_split = feature_role_split  # Geometry/appearance disentanglement flag
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
        # Detail-aware densification: per-Gaussian importance score (EMA-smoothed)
        self.detail_importance = torch.empty(0)
        self._detail_aware_enabled = True  # Will be set by training_setup
        self._detail_ema_decay = 0.9
        self._detail_densify_scale = 0.5
        self._detail_prune_weight = 0.2
        self.optimizer = None
        self.optimizer_i = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
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
        
        # Ensure detail_importance is correctly sized after restore
        # (Old checkpoints may not have this buffer, initialize to neutral)
        if self.detail_importance.shape[0] != self._xyz.shape[0]:
            self.detail_importance = torch.full((self._xyz.shape[0],), 0.5, device="cuda")
            print(f"[INFO] Initialized detail_importance to neutral (0.5) for {self._xyz.shape[0]} Gaussians")
    
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
        self.detail_importance = torch.full((N,), 0.5, device="cuda")
        
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

            # For debug stats (mean norms) - compute only when needed (every 1000 iter)
            shared_norm_sum = 0.0
            geometry_norm_sum = 0.0
            appearance_norm_sum = 0.0
            total_count = 0

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
                    total_count += fused.shape[0]
                    continue

                # Encoder already clamps, skip redundant clamp here (encoder does it)
                # Only clamp if values are out of expected range (rare case)
                
                # Accumulate norms for debug logging (defer .item() to reduce sync)
                chunk_size_actual = shared_chunk.shape[0]
                with torch.no_grad():
                    shared_norm_sum += shared_chunk.norm(dim=-1).sum()
                    geometry_norm_sum += geometry_chunk.norm(dim=-1).sum()
                    appearance_norm_sum += appearance_chunk.norm(dim=-1).sum()
                total_count += chunk_size_actual

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

            # Store mean-norm scalars for light-weight debug logging (convert to float only when needed)
            if total_count > 0:
                self._last_shared_norm = (shared_norm_sum / total_count).item()
                self._last_geometry_norm = (geometry_norm_sum / total_count).item()
                self._last_appearance_norm = (appearance_norm_sum / total_count).item()
            else:
                self._last_shared_norm = 0.0
                self._last_geometry_norm = 0.0
                self._last_appearance_norm = 0.0

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
        
        # NOTE: Removed torch.cuda.empty_cache() here for real-time performance
        # empty_cache() is expensive (~1-2ms) and should only be called when necessary
    
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

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        # explicit attributes
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._sh_mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], self.max_sh_degree), device="cuda").requires_grad_(True))

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
        # Initialize detail_importance to neutral value (0.5) - will be updated via EMA
        self.detail_importance = torch.full((self.get_xyz.shape[0],), 0.5, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        # Detail-aware densification config (from training_args if present)
        self._detail_aware_enabled = getattr(training_args, 'enable_detail_aware', True)
        self._detail_ema_decay = getattr(training_args, 'detail_ema_decay', 0.9)
        self._detail_densify_scale = getattr(training_args, 'detail_densify_scale', 0.5)
        self._detail_prune_weight = getattr(training_args, 'detail_prune_weight', 0.2)
        # Ensure detail_importance is initialized if not already
        if self.detail_importance.shape[0] != self.get_xyz.shape[0]:
            self.detail_importance = torch.full((self.get_xyz.shape[0],), 0.5, device="cuda")
        
        # Update capacity config from training_args (allow CLI override)
        max_gaussians = getattr(training_args, 'max_gaussians', 6_000_000)
        densify_until = getattr(training_args, 'densify_until_iter', 15000)
        prune_interval = getattr(training_args, 'prune_interval', 1000)
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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling_base = optimizable_tensors["scaling_base"]
        self._mask = optimizable_tensors["mask"]
        self._sh_mask = optimizable_tensors["sh_mask"]
        
        # In 3DGS mode, also prune f_rest
        if self._encoder_variant == '3dgs' and "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # Prune detail_importance buffer
        if self.detail_importance.shape[0] > 0:
            self.detail_importance = self.detail_importance[valid_points_mask]
        
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
        
        # Extend detail_importance for new Gaussians: inherit from mean (neutral bias)
        num_new = new_xyz.shape[0]
        if num_new > 0:
            # New Gaussians start with mean importance (will converge via EMA)
            mean_importance = self.detail_importance.mean().item() if self.detail_importance.numel() > 0 else 0.5
            new_importance = torch.full((num_new,), mean_importance, device="cuda")
            self.detail_importance = torch.cat([self.detail_importance, new_importance], dim=0)
        
        # Invalidate attribute caches; shapes changed
        self._cached_attributes_hash = None
        if hasattr(self, '_cached_contracted_key'):
            self._cached_contracted_key = None

    def compute_densify_score(self, grads, strategy="feature_weighted"):
        """
        Compute densification scores for ablation studies.
        
        Args:
            grads: Gradient tensor [N, 3] or [N]
            strategy: Densification strategy variant
                - "original_3dgs": Original 3DGS gradient-only baseline
                - "feature_weighted": GlowGS feature-weighted densification
        
        Returns:
            scores: Densification scores [N], higher = more likely to densify
        """
        n_points = grads.shape[0]
        
        # Normalize gradient magnitude to [0, 1] range for comparability
        grad_norm = torch.norm(grads, dim=-1) if grads.dim() > 1 else grads
        grad_max = grad_norm.max()
        if grad_max > 0:
            grad_score = grad_norm / grad_max
        else:
            grad_score = grad_norm
        
        if strategy == "original_3dgs":
            # 3DGS baseline: pure gradient magnitude (no detail_importance)
            return grad_score
        
        elif strategy == "feature_weighted":
            # GlowGS: gradient modulated by detail_importance
            if self._detail_aware_enabled and self.detail_importance.shape[0] >= n_points:
                k = self._detail_densify_scale
                # Higher detail_importance → higher score (easier to densify)
                importance_boost = 1.0 + k * self.detail_importance[:n_points]
                return grad_score * importance_boost
            else:
                # Fallback to original if detail_importance not available
                return grad_score
        
        else:
            raise ValueError(f"Unknown densify_strategy: {strategy}. Expected 'original_3dgs' or 'feature_weighted'")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        Feature-weighted densification: split large Gaussians (GlowGS innovation #3).
        
        Key Modification:
        - Applies per-Gaussian detail_importance scores to modulate split threshold.
        - High-importance Gaussians (edges, textures) get lower effective threshold,
          making them easier to split even with moderate gradients.
        - Reallocates limited Gaussian budget toward high-frequency regions.
        
        Formula: effective_threshold_i = base_threshold / (1 + k * detail_importance_i)
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # Detail-aware threshold: high detail_importance → lower effective threshold
        # effective_threshold_i = grad_threshold / (1 + k * detail_importance_i)
        if self._densify_strategy == "feature_weighted" and self._detail_aware_enabled and self.detail_importance.shape[0] == n_init_points:
            k = self._detail_densify_scale
            effective_threshold = grad_threshold / (1.0 + k * self.detail_importance)
        else:
            # original_3dgs or fallback: use fixed threshold
            effective_threshold = grad_threshold
        
        selected_pts_mask = padded_grad >= effective_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_scaling_base = self.scaling_base_inverse_activation(self.get_scaling_base[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_mask = self._mask[selected_pts_mask].repeat(N,1)
        new_sh_mask = self._sh_mask[selected_pts_mask].repeat(N,1)
        
        # In 3DGS mode, also clone/split f_rest
        new_features_rest = None
        if self._encoder_variant == '3dgs' and isinstance(self._features_rest, nn.Parameter):
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        n_points = grads.shape[0]
        
        # Detail-aware threshold: high detail_importance → lower effective threshold
        if self._densify_strategy == "feature_weighted" and self._detail_aware_enabled and self.detail_importance.shape[0] >= n_points:
            k = self._detail_densify_scale
            effective_threshold = grad_threshold / (1.0 + k * self.detail_importance[:n_points])
            selected_pts_mask = torch.norm(grads, dim=-1) >= effective_threshold
        else:
            # original_3dgs or fallback: use fixed threshold
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_scaling_base = self._scaling_base[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_sh_mask = self._sh_mask[selected_pts_mask]
        
        # In 3DGS mode, also clone f_rest
        new_features_rest = None
        if self._encoder_variant == '3dgs' and isinstance(self._features_rest, nn.Parameter):
            new_features_rest = self._features_rest[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask, new_features_rest)
        # Recompute implicit attributes to align shapes before further densification logic
        self.update_attributes(force_update=True)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Optimization: Only update once before densification
        # Cache will be invalidated after densify_and_clone/split modify _xyz
        self.update_attributes(force_update=True)
        self.densify_and_clone(grads, max_grad, extent)
        # Cache invalidated by densify_and_clone, will auto-update on next call
        self.densify_and_split(grads, max_grad, extent)

        # Final update before pruning (force to ensure we have latest attributes)
        self.update_attributes(force_update=True)
        
        # Base prune conditions: LocoGS uses fixed mask_threshold=0.01, min_opacity from train.py (0.005)
        mask_condition = (self.get_mask <= 0.01).squeeze()
        opacity_condition = (self.get_opacity < min_opacity).squeeze()
        prune_mask = torch.logical_or(mask_condition, opacity_condition)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # Detail-aware pruning: among candidates, prefer pruning low detail_importance ones
        # This is a soft bias: we don't override the base conditions, but when capacity
        # is tight, low-detail Gaussians are more likely to be pruned
        if self._detail_aware_enabled and self.detail_importance.shape[0] == self.get_xyz.shape[0]:
            # Compute prune score: higher score → more likely to prune
            # Base score from opacity (lower opacity → higher prune score)
            opacity_score = 1.0 - self.get_opacity.squeeze()
            # Detail score: low detail_importance → higher prune score
            detail_score = 1.0 - self.detail_importance
            
            # Combined score with weighting
            w_detail = self._detail_prune_weight
            combined_score = (1.0 - w_detail) * opacity_score + w_detail * detail_score
            
            # Apply detail-based pruning only to borderline cases
            # Gaussians that barely pass the opacity threshold but have low detail_importance
            borderline_mask = torch.logical_and(
                self.get_opacity.squeeze() >= min_opacity,
                self.get_opacity.squeeze() < min_opacity * 2.0  # Within 2x of threshold
            )
            # Among borderline cases, prune those with very low detail_importance
            low_detail_threshold = 0.15  # Bottom 15% of importance range
            borderline_low_detail = torch.logical_and(
                borderline_mask,
                self.detail_importance < low_detail_threshold
            )
            # Add these to prune mask (soft pruning of low-detail borderline cases)
            prune_mask = torch.logical_or(prune_mask, borderline_low_detail)
        
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

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
        Capacity-controlled densification wrapper (GlowGS innovation #3: N_max enforcement).
        
        Design:
        - Enforces hard point budget (N_max, default 6M) to prevent memory explosion.
        - Densification only proceeds if: (1) within schedule, (2) under capacity.
        - Works synergistically with feature-weighted densification: limited budget
          is allocated to high-importance regions via detail_importance modulation.
        - Post-densification: periodic mask pruning to remove low-contribution Gaussians.
        
        Does NOT modify core densify/prune algorithms, only adds capacity gating.
        
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
        
        # Densify only if under capacity and within schedule
        if iteration < densify_until and num_points < max_points:
            self.densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
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

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_detail_importance(self, pixel_importance, visibility_filter, radii):
        """
        Update per-Gaussian detail_importance via EMA using pixel-level importance.
        
        Uses a lightweight approximation: for visible Gaussians, sample the mean
        pixel_importance in a small region around their projected screen position.
        This avoids expensive per-pixel-per-Gaussian weight accumulation.
        
        Args:
            pixel_importance: [H, W] tensor with values in [0, 1]
            visibility_filter: [N] boolean mask of visible Gaussians
            radii: [N] screen-space radii of Gaussians (used as proxy for contribution extent)
        """
        if not self._detail_aware_enabled:
            return
        
        N = self.get_xyz.shape[0]
        if N == 0 or pixel_importance is None:
            return
        
        # Ensure detail_importance is correctly sized
        if self.detail_importance.shape[0] != N:
            self.detail_importance = torch.full((N,), 0.5, device="cuda")
        
        # For visible Gaussians, estimate their importance from pixel_importance
        # Simple approximation: use spatial mean of pixel_importance as proxy
        # (More accurate would require splatting weights, but too expensive)
        visible_mask = visibility_filter
        num_visible = visible_mask.sum().item()
        
        if num_visible == 0:
            return
        
        # Compute mean pixel importance as a global batch statistic
        # Visible Gaussians inherit this weighted by their screen coverage
        mean_pixel_importance = pixel_importance.mean().item()
        
        # Radii-weighted importance: larger screen footprint → more contribution
        # Normalize radii to [0, 1] range for weighting
        radii_visible = radii[visible_mask].float()
        if radii_visible.numel() > 0:
            radii_norm = radii_visible / (radii_visible.max().clamp(min=1.0))
            # Gaussians with larger radii contribute more to image → weight them higher
            # but also cap influence to avoid single large Gaussian dominating
            weight = (0.3 + 0.7 * radii_norm.clamp(0.0, 1.0))
        else:
            weight = torch.ones(num_visible, device="cuda")
        
        # Sample-based approximation: each visible Gaussian gets importance
        # proportional to mean_pixel_importance * its coverage weight
        tmp_importance = torch.full((N,), 0.0, device="cuda")
        tmp_importance[visible_mask] = mean_pixel_importance * weight
        
        # EMA update: smooth over iterations to avoid single-frame noise
        ema_decay = self._detail_ema_decay
        # Only update visible Gaussians; others retain their current importance
        self.detail_importance[visible_mask] = (
            ema_decay * self.detail_importance[visible_mask] +
            (1.0 - ema_decay) * tmp_importance[visible_mask]
        )
        
        # Clamp to valid range
        self.detail_importance.clamp_(0.0, 1.0)

    def get_detail_importance_stats(self):
        """Return percentile statistics for logging."""
        if self.detail_importance.numel() == 0:
            return {'p5': 0.0, 'p50': 0.0, 'p95': 0.0}
        di = self.detail_importance
        return {
            'p5': torch.quantile(di, 0.05).item(),
            'p50': torch.quantile(di, 0.50).item(),
            'p95': torch.quantile(di, 0.95).item(),
        }

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
        # Pack all grid components into a single compressed archive to reduce header
        # overhead and improve compression across parameters.
        grid_save_dict = {}
        for key, value in grid_state_dict.items():
            normalized_key = key.replace('.', '_')  # e.g., "hash_encoder.params" -> "hash_encoder_params"
            param_data = value.detach().cpu().numpy()
            # Quantize grid parameters (log quantization) to uint8
            param_quant, param_scale, param_min = quantize(param_data, bit=6, log=True)
            grid_save_dict[f"{normalized_key}_data"] = param_quant.astype(np.uint8)
            grid_save_dict[f"{normalized_key}_scale"] = param_scale
            grid_save_dict[f"{normalized_key}_min"] = param_min
            grid_save_dict[f"{normalized_key}_orig"] = np.array(key)

        # Save single combined file
        np.savez_compressed(os.path.join(path, "grid_all.npz"), **grid_save_dict)

        # G-PCC encoding
        encode_xyz(xyz, path)

        # uniform quantization
        f_dc_quant, f_dc_scale, f_dc_min = quantize(f_dc, bit=8)
        scaling_base_quant, scaling_base_scale, scaling_base_min = quantize(scale_base, bit=8)

        np.savez_compressed(os.path.join(path, "f_dc.npz"), data=f_dc_quant.astype(np.uint8), scale=f_dc_scale, min=f_dc_min)
        np.savez_compressed(os.path.join(path, "scale_base.npz"), data=scaling_base_quant.astype(np.uint8), scale=scaling_base_scale, min=scaling_base_min)
        
        # sh mask
        sh_mask = (sh_mask > 0.01).cumprod(axis=-1).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "sh_mask.npz"), data=sh_mask)

        # mlps: quantize MLP parameter tensors to 8-bit (log) and pack into one file
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
            self._grid.load_state_dict(grid_state_dict, strict=True)

        # sh_mask
        sh_mask = np.load(os.path.join(path, 'sh_mask.npz'))['data'].astype(np.float32)
        self._sh_mask = torch.from_numpy(np.asarray(sh_mask)).cuda()

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
            o = dequantize(mlp_npz['opacity_data'], mlp_npz['opacity_scale'], mlp_npz['opacity_min'], log=True)
            fr = dequantize(mlp_npz['f_rest_data'], mlp_npz['f_rest_scale'], mlp_npz['f_rest_min'], log=True)
            sc = dequantize(mlp_npz['scale_data'], mlp_npz['scale_scale'], mlp_npz['scale_min'], log=True)
            rt = dequantize(mlp_npz['rotation_data'], mlp_npz['rotation_scale'], mlp_npz['rotation_min'], log=True)

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
        self.update_attributes()
        sh_mask = self._sh_mask
        sh_mask_degree = torch.ones_like(self._features_rest) # (N, 15, 3)
        for deg in range(1, self.active_sh_degree + 1):
            sh_mask_degree[:, deg**2 - 1:, :] *= sh_mask[:, deg - 1:deg].unsqueeze(1)
        
        self._features_rest *= sh_mask_degree
        self.sh_levels = sh_mask.sum(1).int()
