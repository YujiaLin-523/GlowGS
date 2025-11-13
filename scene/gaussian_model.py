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
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, contract_to_unisphere, mortonEncode
from utils.quantization_utils import half_ste, quantize, dequantize
from utils.gpcc_utils import encode_xyz, decode_xyz

import tinycudann as tcnn

# [HYBRID] Import for hybrid encoder feature
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from encoders.vm_encoder import VMEncoder
import torch.nn.functional as F


# ============================================================================
# HYBRID FEATURE ENCODER (VM + HashGrid Fusion)
# ============================================================================
class HybridFeatureEncoder(torch.nn.Module):
    """
    Hybrid encoder that fuses VM (low-frequency) and HashGrid (high-frequency) features.
    
    Architecture:
        - VM branch: Low-rank tri-plane encoder for global structure
        - Hash branch: Existing HashGrid encoder for local details (unchanged)
        - Fusion: Residual combination weighted by scale-based frequency gate
    
    Fusion formula:
        out = proj_vm(vm_feat) + hash_gain * m_hf * proj_hash(hash_feat)
        
    Where m_hf is a per-point high-frequency gate in [0,1] derived from Gaussian scale:
        - Smaller Gaussians (fine details) => higher m_hf => more HashGrid influence
        - Larger Gaussians (coarse structure) => lower m_hf => more VM influence
    
    Args:
        hash_encoder: Existing HashGrid encoder (tcnn.Encoding)
        out_dim: Output feature dimension (must match downstream expectations)
        config: HybridEncoderParams object with configuration
    
    Invariants:
        - Output shape matches original encoder: [N, out_dim]
        - Hash encoder is NOT modified, only wrapped
        - Safe to disable via config without code changes
    """
    
    def __init__(self, hash_encoder: torch.nn.Module, out_dim: int, config):
        super().__init__()
        
        # Store config
        self.config = config
        
        # Wrap existing hash encoder (no modifications)
        self.hash = hash_encoder
        
        # Infer hash output dimension by dry-run (safe introspection)
        # This avoids hard-coding dimension assumptions
        sample_xyz = torch.zeros(1, 3, device='cuda')
        try:
            with torch.no_grad():
                hash_dim = self.hash(sample_xyz).shape[-1]
        except Exception:
            # Fallback: assume 32 if introspection fails (conservative default)
            hash_dim = 32
            print(f"[HYBRID] Warning: Could not infer hash dim, using default {hash_dim}")
        
        # VM encoder for low-frequency features
        self.vm = VMEncoder(
            in_dim=3,
            rank=config.vm_rank,
            plane_res=config.vm_plane_res,
            out_dim=out_dim,  # VM outputs directly to out_dim
            basis=config.vm_basis,
            use_checkpoint=config.vm_checkpoint,
            lpf_enable=config.vm_lpf_enable,
            lpf_kernel=config.vm_lpf_kernel
        )
        
        # [CHECKLIST 6] Enable tri-plane rotation perturbation if configured
        # This is set at training time via hybrid_config, not here
        self.vm.rotation_perturb_enable = False
        self.vm.rotation_perturb_angle = 3.0
        
        # Projection layers to align dimensions
        # Hash needs projection from hash_dim to out_dim
        self.proj_hash = torch.nn.Linear(hash_dim, out_dim, bias=False)
        
        # VM already outputs out_dim, so identity projection
        self.proj_vm = torch.nn.Identity()
        
        # [NEW] VM auxiliary head for low-frequency proxy loss (created at init to avoid torch.compile issues)
        self.head_vm = torch.nn.Linear(out_dim, 3, bias=False)
        torch.nn.init.xavier_uniform_(self.head_vm.weight, gain=0.5)
        
        # [FIX] Hash gain starts small (warm-up via runtime setter)
        # [ORIG] self.register_buffer('hash_gain', torch.tensor(0.5))
        self.register_buffer('hash_gain', torch.tensor(config.init_hash_gain))  # Start at 0.05
        
        # [FIX] Fusion floor parameter (beta_min) for convex blend to ensure VM always gets gradients
        # beta = beta_min + (1 - 2*beta_min) * beta_raw ensures both branches receive gradients
        self.beta_min = config.beta_min_start  # Will be annealed by trainer
        
        # [FIX] Feature scaling kept safe (constant 1.0 or trainable with clamp)
        # [ORIG] self.feature_scale = torch.nn.Parameter(torch.tensor(15.0))
        if config.feature_scale_trainable:
            self.feature_scale = torch.nn.Parameter(torch.tensor(config.feature_scale_init))
            self.feature_scale_trainable = True
        else:
            self.register_buffer('feature_scale', torch.tensor(config.feature_scale_init))
            self.feature_scale_trainable = False
        
        # [FIX] Gate schedule parameters (annealed at runtime)
        # [ORIG] gate_alpha/tau were static config values
        self.gate_alpha = config.gate_alpha_start  # Will be updated by trainer
        self.gate_tau = config.gate_tau_start      # Will be updated by trainer
        
        # Initialize projection weights with smaller gain to prevent instability
        torch.nn.init.xavier_uniform_(self.proj_hash.weight, gain=0.5)
        
        # [HYBRID] Expose n_output_dims for compatibility with downstream code
        self.n_output_dims = out_dim
    
    @staticmethod
    def _scale_to_gate_impl(sigma_max: torch.Tensor, gate_alpha: float, gate_tau: float, 
                            gate_kappa: float, gate_clamp: tuple) -> torch.Tensor:
        """
        Compute high-frequency gate from Gaussian maximum scale.
        
        Uses sigmoid to smoothly map scale to frequency weight:
            m_hf = sigmoid(alpha * (log2(kappa / sigma_max) - tau))
        
        Intuition:
            - Small sigma (fine Gaussians) => large log2(kappa/sigma) => m_hf → 1
            - Large sigma (coarse Gaussians) => small log2(kappa/sigma) => m_hf → 0
        
        Args:
            sigma_max: [N] per-Gaussian maximum scale (anisotropic scale's max dimension)
            gate_alpha, gate_tau, gate_kappa, gate_clamp: config parameters
        
        Returns:
            [N] high-frequency gate values in [gate_clamp[0], gate_clamp[1]]
        """
        eps = 1e-8  # Numerical stability for log
        
        # Log2-space scale comparison
        x = torch.log2(gate_kappa / (sigma_max + eps))
        
        # Sigmoid with configurable slope and threshold
        m = torch.sigmoid(gate_alpha * (x - gate_tau))
        
        # Safety clamp to configured range
        lo, hi = gate_clamp
        return torch.clamp(m, lo, hi)
    
    def _scale_to_gate(self, sigma_max: torch.Tensor) -> torch.Tensor:
        """
        Wrapper that uses runtime-annealed gate parameters.
        [FIX] Uses self.gate_alpha/tau (updated by trainer) instead of static config.
        """
        return self._scale_to_gate_impl(
            sigma_max, 
            self.gate_alpha,  # Runtime value (annealed)
            self.gate_tau,    # Runtime value (annealed)
            self.config.gate_kappa,
            self.config.gate_clamp
        )
    
    def set_hash_gain(self, value: float):
        """
        Runtime setter for hash_gain (warm-up schedule).
        [FIX] Allows trainer to gradually increase hash influence from 0.05 → 1.0.
        """
        self.hash_gain.fill_(float(value))
    
    def set_gate_sched(self, alpha: float, tau: float):
        """
        Set gate schedule parameters at runtime (annealing).
        [FIX] Allows trainer to anneal alpha (3→8) and tau (1→0) over warm-up period.
        """
        self.gate_alpha = float(alpha)
        self.gate_tau = float(tau)
    
    def set_beta_min(self, beta_min: float):
        """
        Set fusion floor parameter at runtime (annealing).
        [FIX] Allows trainer to anneal beta_min from 0.10 → 0.05 over warm-up period.
        """
        self.beta_min = float(beta_min)
    
    def forward(self, xyz: torch.Tensor, sigma_max: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: fuse VM and HashGrid features.
        
        Args:
            xyz: [N, 3] normalized coordinates in [0,1]^3 (contracted space)
            sigma_max: [N] per-Gaussian maximum scale for frequency gating (optional)
                      If None, uses default gate value (no scale-based adaptation)
        
        Returns:
            [N, out_dim] fused features (same shape as original encoder output)
        """
        # [FIX A-1] Unify the spatial domain to [0,1]^3 for both branches.
        # This aligns neighborhood semantics and stabilizes gradients.
        # Coordinate sanitization: handle NaN/Inf and clamp to [0,1]
        xyz01 = torch.nan_to_num(xyz, nan=0.5, posinf=0.5, neginf=0.5).clamp(0.0, 1.0)
        
        # Hash branch (existing encoder, unchanged API)
        # tcnn outputs half precision, convert to float for consistency
        h = self.hash(xyz01)            # [N, hash_dim], dtype=half
        h = h.float()                   # Convert to float32
        h = self.proj_hash(h)           # [N, out_dim]
        
        # VM branch
        v = self.vm(xyz01)              # [N, out_dim], dtype=float32
        
        # [FIX A-1] Ephemeral feature cache for VM aux loss (kept in the graph).
        # Must be cleared right after loss/backward to avoid graph growth.
        if self.training:
            self._vm_feat_for_aux = v  # No detach - need gradients for aux loss
            self._last_vm_feat = v.detach()    # For debug only
            self._last_hash_feat = h.detach()  # For debug only
        else:
            # Inference mode: don't cache to save memory
            self._vm_feat_for_aux = None
            self._last_vm_feat = None
            self._last_hash_feat = None
        
        # [FIX A-1] Compute frequency gate from scale with numeric safeguards.
        # Use single-sigmoid mapping from _scale_to_gate to avoid double compression of gradients.
        if sigma_max is not None:
            # Sanitize sigma_max to prevent NaN/Inf in log operations
            sigma_max = torch.nan_to_num(sigma_max, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-6)
            m_hf = self._scale_to_gate(sigma_max)  # [N], already sigmoid output
        else:
            # Default: assume medium-scale Gaussians (neutral gate)
            m_hf = torch.ones(xyz01.shape[0], device=xyz01.device) * 0.5
        
        # Clamp gate output to valid range
        m_hf = m_hf.clamp(self.config.gate_clamp[0], self.config.gate_clamp[1])  # [N]
        
        # [FIX] Floored convex blend ensures BOTH branches always receive gradients
        # [ORIG] out = v + (self.hash_gain * m_hf) * h  (residual: VM gradients killed when hash dominates)
        # Floored convex: beta = beta_min + (1 - 2*beta_min) * beta_raw
        # This ensures beta ∈ [beta_min, 1-beta_min], guaranteeing VM weight ≥ beta_min
        beta_raw = (self.hash_gain * m_hf).unsqueeze(-1)  # [N, 1], raw blend weight 0..1
        beta = self.beta_min + (1.0 - 2.0 * self.beta_min) * beta_raw.clamp(0.0, 1.0)  # [N, 1], floored
        
        out = (1.0 - beta) * v + beta * h  # Convex combination
        
        # [FIX] Feature scaling kept safe (constant 1.0, trainable removed to prevent instability)
        # [ORIG] out = out * self.feature_scale (could grow unbounded if trainable)
        scale = self.feature_scale
        out = out * scale
        
        # [FIX] Final NaN guard before returning
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # ========================================
        # [DEBUG] Cache intermediate values for gradient diagnostics
        # These are read-only references, do NOT detach - must stay in computation graph
        # ========================================
        try:
            from arguments import DEBUG_CFG
            if DEBUG_CFG.ON:
                self._dbg_vm = v              # [N, out_dim] VM features (in graph)
                self._dbg_hash = h            # [N, out_dim] Hash features (in graph)
                self._dbg_beta = beta         # [N, 1] floored blend weight (in graph)
                self._dbg_gate = m_hf         # [N] gate after sigmoid (in graph)
        except:
            pass  # If DEBUG_CFG not imported, silently skip
        
        return out


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
        # [ORIG] Original HashGrid encoder setup
        # self._grid = tcnn.Encoding(n_input_dims=3, encoding_config=self.encoding_config)
        
        # [HYBRID] Conditional encoder setup based on config
        if self.hybrid_config and self.hybrid_config.hybrid_enable:
            # Build original hash encoder first
            hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=self.encoding_config)
            
            # Wrap with hybrid encoder (VM + Hash fusion)
            # Output dim inferred from grid encoding output
            grid_out_dim = hash_encoder.n_output_dims
            self._grid = HybridFeatureEncoder(hash_encoder, out_dim=grid_out_dim, config=self.hybrid_config).cuda()
            
            print(f"[HYBRID] Enabled: VM(rank={self.hybrid_config.vm_rank}, res={self.hybrid_config.vm_plane_res}) + HashGrid")
        else:
            # Original path: HashGrid only
            self._grid = tcnn.Encoding(n_input_dims=3, encoding_config=self.encoding_config)
        
        # Rest of the network heads remain unchanged (they consume grid features)
        self._features_rest_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=(self.max_sh_degree + 1) ** 2 * 3 - 3, network_config=self.network_config)
        self._scaling_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=3, network_config=self.network_config)
        self._rotation_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=4, network_config=self.network_config)
        self._opacity_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=1, network_config=self.network_config)
        
        self._aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float, device='cuda')
        self._rotation_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device='cuda')
        self._opacity_init = torch.tensor([[np.log(0.1 / (1 - 0.1))]], dtype=torch.float, device='cuda')

    def __init__(self, sh_degree : int, hash_size = 19, width = 64, depth = 2, hybrid_config=None, aniso_config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling_base = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._mask = torch.empty(0).cuda()
        self._sh_mask = torch.empty(0).cuda()
        self._grid = torch.empty(0).cuda()
        self._features_rest_head = torch.empty(0).cuda()
        self._scaling_head = torch.empty(0).cuda()
        self._rotation_head = torch.empty(0).cuda()
        self._opacity_head = torch.empty(0).cuda()
        self._aabb = torch.empty(0).cuda()
        self._rotation_init = torch.empty(0).cuda()
        self._opacity_init = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.optimizer = None
        self.optimizer_i = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.hybrid_config = hybrid_config  # Store for later use
        self.aniso_config = aniso_config    # [NEW] Store anisotropy constraint config (CHECKLIST 2)
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
    
    def restore(self, model_args, training_args):
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

    def update_attributes(self):
        # [ORIG] Original encoder call:
        # feats = self._grid(self.get_contracted_xyz)
        
        # [HYBRID] Conditional encoder call with scale-based gating
        # Use get_contracted_xyz which has detach() to match original behavior
        contracted_xyz = self.get_contracted_xyz
        N = contracted_xyz.shape[0]
        
        # [FIX D] Encoder micro-batching: only when memory is constrained
        # Only batch if N exceeds BOTH batch_size AND oom_threshold
        # This allows full-batch training when memory is sufficient
        micro_batch_size = 0
        oom_threshold = float('inf')  # Default: never batch
        
        if self.hybrid_config and hasattr(self.hybrid_config, 'update_batch_size'):
            micro_batch_size = self.hybrid_config.update_batch_size
            if hasattr(self.hybrid_config, 'oom_threshold'):
                oom_threshold = self.hybrid_config.oom_threshold
        
        # Only use micro-batching if both conditions met:
        # 1. N exceeds OOM threshold (memory constraint)
        # 2. Batch size is configured (micro_batch_size > 0)
        use_micro_batch = (micro_batch_size > 0 and N > oom_threshold)
        
        if use_micro_batch:
            # [ACCEPTANCE CHECK D] Log micro-batch usage (first time only to avoid spam)
            if not hasattr(self, '_logged_microbatch'):
                print(f"[MICRO-BATCH] Memory-constrained mode activated: N={N} > threshold={oom_threshold}")
                print(f"[MICRO-BATCH] Using chunked forward: batch_size={micro_batch_size}, "
                      f"num_batches={(N + micro_batch_size - 1) // micro_batch_size}")
                self._logged_microbatch = True
            
            # Process encoder in chunks to save VRAM
            feat_list = []
            num_batches = (N + micro_batch_size - 1) // micro_batch_size
            
            for i in range(num_batches):
                start_idx = i * micro_batch_size
                end_idx = min((i + 1) * micro_batch_size, N)
                xyz_chunk = contracted_xyz[start_idx:end_idx]
                
                if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
                    # Hybrid mode: pass scale for frequency gating
                    current_N = contracted_xyz.shape[0]
                    if (self._scaling.numel() > 0 and 
                        self._scaling_base.numel() > 0 and 
                        self._scaling.shape[0] == current_N):
                        sigma_max_chunk = self.get_scaling[start_idx:end_idx].max(dim=-1).values.detach()
                        feats_chunk = self._grid(xyz_chunk, sigma_max=sigma_max_chunk)
                    else:
                        feats_chunk = self._grid(xyz_chunk)
                else:
                    # Original mode: standard encoder call
                    feats_chunk = self._grid(xyz_chunk)
                
                feat_list.append(feats_chunk)
            
            feats = torch.cat(feat_list, dim=0)
        else:
            # Normal path: process all points at once
            if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
                # Hybrid mode: pass scale for frequency gating
                # Use maximum anisotropic scale as frequency proxy
                # Check if scaling tensors have been initialized AND sizes match current points
                # (after densify/prune operations, _scaling may be stale while _scaling_base is updated)
                current_N = contracted_xyz.shape[0]
                if (self._scaling.numel() > 0 and 
                    self._scaling_base.numel() > 0 and 
                    self._scaling.shape[0] == current_N):
                    # Detach sigma_max to prevent graph reuse error
                    # sigma_max is only used for gating weight computation, not for gradients
                    sigma_max = self.get_scaling.max(dim=-1).values.detach()  # [N]
                    feats = self._grid(contracted_xyz, sigma_max=sigma_max)
                else:
                    # Fallback: skip sigma_max if sizes mismatch (during densify/prune)
                    # or during initialization phase
                    feats = self._grid(contracted_xyz)
            else:
                # Original mode: standard encoder call
                feats = self._grid(contracted_xyz)
        
        # Rest of attribute updates unchanged
        self._opacity = self._opacity_head(feats).float() + self._opacity_init
        self._scaling = self._scaling_head(feats).float()
        self._rotation = self._rotation_head(feats).float() + self._rotation_init
        if self.max_sh_degree > 0:
            self._features_rest = self._features_rest_head(feats).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3).float()
        
        torch.cuda.empty_cache()

    @property
    def get_scaling_base(self):
        return self.scaling_base_activation(self._scaling_base)

    @property
    def get_scaling(self):
        return self.get_scaling_base * self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_aniso_clamp(self):
        """
        Get scaled Gaussians with anisotropy hard-clamp applied.
        
        Clamps aspect ratio r = max(s)/min(s) to r_max to prevent paint-roller artifacts.
        Only modifies internal _scaling_raw before activation (no external API change).
        
        Returns:
            [N, 3] scaling with aspect ratio clamped
        """
        if not self.aniso_config or not self.aniso_config.aniso_constraint_enable:
            return self.get_scaling
        
        # Get raw scaling (before activation)
        scaling_raw = self.get_scaling  # [N, 3]
        
        # Compute aspect ratio for each Gaussian
        s_min = scaling_raw.min(dim=-1)[0]  # [N]
        s_max = scaling_raw.max(dim=-1)[0]  # [N]
        aspect_ratio = s_max / (s_min.clamp(min=1e-8))  # [N]
        
        # Hard clamp: if r > r_max, scale down the max dimensions
        r_max = self.aniso_config.aniso_r_max
        scale_factor = torch.ones_like(aspect_ratio)
        exceed_mask = aspect_ratio > r_max
        scale_factor[exceed_mask] = (s_max / (aspect_ratio * s_min)).clamp(max=1.0)[exceed_mask]
        
        # Apply clamping
        scaling_clamped = scaling_raw * scale_factor.unsqueeze(-1)
        return scaling_clamped
    
    def compute_aniso_regularizer(self, iteration: int) -> torch.Tensor:
        """
        Compute soft regularizer for anisotropy: lambda_aniso * ReLU(r - r_t)^2
        
        Anneals from aniso_lambda (5k iter) to 0 (15k iter).
        
        Args:
            iteration: Current training iteration
        
        Returns:
            Scalar loss tensor
        """
        if not self.aniso_config or not self.aniso_config.aniso_constraint_enable:
            return torch.tensor(0.0, device=self.get_scaling.device)
        
        scaling = self.get_scaling  # [N, 3]
        s_min = scaling.min(dim=-1)[0]  # [N]
        s_max = scaling.max(dim=-1)[0]  # [N]
        aspect_ratio = s_max / (s_min.clamp(min=1e-8))  # [N]
        
        # Anneal regularizer weight from start to end iteration
        start = self.aniso_config.aniso_lambda_anneal_start
        end = self.aniso_config.aniso_lambda_anneal_end
        if iteration < start:
            lambda_w = 0.0
        elif iteration > end:
            lambda_w = 0.0
        else:
            # Linear decay: 1.0 at start, 0.0 at end
            progress = (iteration - start) / (end - start)
            lambda_w = 1.0 - progress  # Decaying from 1 to 0
        
        # Soft penalty: only on aspect ratios exceeding threshold
        r_t = self.aniso_config.aniso_r_threshold
        excess = F.relu(aspect_ratio - r_t)  # [N]
        loss = (excess ** 2).mean()
        
        return lambda_w * self.aniso_config.aniso_lambda * loss
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self.xyz_activation(self._xyz)

    @property
    def get_contracted_xyz(self):
        return contract_to_unisphere(self.get_xyz.detach(), self._aabb)
        
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
        """
        Initialize Gaussians from point cloud (LocoGS original: no downsampling).
        
        Args:
            pcd: Input point cloud with points, colors, normals
            spatial_lr_scale: Spatial learning rate scale factor
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # Use full SfM point cloud for healthy densification growth
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        # explicit attributes
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._sh_mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], self.max_sh_degree), device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_dc_lr, "name": "f_dc"},
            {'params': [self._scaling_base], 'lr': training_args.scaling_base_lr, "name": "scaling_base"},
            {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
            {'params': [self._sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        # [HYBRID ENCODER PARAMS] Conditional parameter groups for hybrid and original encoder
        param_groups_i = []
        
        if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
            # [FIX] Hybrid mode: use optimizer param groups with higher LR for VM, no weight decay
            # [ORIG] Manual grad scaling removed; optimizer handles VM learning rate boost
            
            # Hash branch (wrapped inside hybrid encoder) - base learning rate
            param_groups_i.append({
                'params': self._grid.hash.parameters(), 
                'lr': 0.0, 
                'weight_decay': 0.0,
                "name": "grid"
            })
            
            # VM encoder: unified param group with boosted LR and no weight decay
            # VM planes have inherently weak gradients -> need higher LR multiplier
            vm_params = [self._grid.vm.xy, self._grid.vm.xz, self._grid.vm.yz]
            vm_params.extend(self._grid.vm.mlp.parameters())
            param_groups_i.append({
                'params': vm_params, 
                'lr': 0.0,  # Will be set to base_lr * vm_lr_multiplier in update_learning_rate
                'weight_decay': 0.0,  # No weight decay preserves VM capacity
                "name": "vm_encoder"
            })
            
            # Projection layers - base learning rate
            param_groups_i.append({
                'params': self._grid.proj_hash.parameters(), 
                'lr': 0.0, 
                'weight_decay': 0.0,
                "name": "proj_hash"
            })
            
            # Feature scale parameter (if trainable)
            if self._grid.feature_scale_trainable:
                param_groups_i.append({
                    'params': [self._grid.feature_scale], 
                    'lr': 0.0, 
                    'weight_decay': 0.0,
                    "name": "feature_scale"
                })
        else:
            # Original mode: single grid param group
            param_groups_i.append({
                'params': self._grid.parameters(), 
                'lr': 0.0, 
                "name": "grid"
            })
        
        # Rest of the network heads (unchanged)
        param_groups_i.extend([
            {'params': list(self._opacity_head.parameters()), 'lr': 0.0, "name": "opacity"},
            {'params': self._features_rest_head.parameters(), 'lr': 0.0, "name": "f_rest"},
            {'params': self._scaling_head.parameters(), 'lr': 0.0, "name": "scaling"},
            {'params': list(self._rotation_head.parameters()), 'lr': 0.0, "name": "rotation"},
        ])
        
        self.optimizer_i = torch.optim.Adam(param_groups_i, lr=0.0, eps=1e-15)
        
        # ========================================
        # [DEBUG] Optimizer parameter group validation
        # ========================================
        try:
            from arguments import DEBUG_CFG
            if DEBUG_CFG.ON:
                total_params = sum(len(list(g['params'])) for g in param_groups_i)
                print(f"[DEBUG-OPTIMIZER] Total param_groups in optimizer_i: {len(param_groups_i)}, total params: {total_params}")
                
                if self.hybrid_config and self.hybrid_config.hybrid_enable:
                    vm_group = next((g for g in param_groups_i if g['name'] == 'vm_encoder'), None)
                    if vm_group:
                        vm_param_count = len(list(vm_group['params']))
                        print(f"[DEBUG-OPTIMIZER] VM encoder group found: {vm_param_count} parameters")
                        
                        # Verify VM parameters are actually in the group
                        vm_actual = [self._grid.vm.xy, self._grid.vm.xz, self._grid.vm.yz]
                        vm_actual.extend(self._grid.vm.mlp.parameters())
                        vm_actual_ids = set(id(p) for p in vm_actual)
                        vm_group_ids = set(id(p) for p in vm_group['params'])
                        
                        if vm_actual_ids == vm_group_ids:
                            print(f"[DEBUG-OPTIMIZER] ✓ VM parameters correctly registered in optimizer")
                        else:
                            missing = len(vm_actual_ids - vm_group_ids)
                            extra = len(vm_group_ids - vm_actual_ids)
                            print(f"[DEBUG-OPTIMIZER] ✗ VM parameter mismatch: {missing} missing, {extra} extra")
                    else:
                        print(f"[DEBUG-OPTIMIZER] ✗ WARNING: VM encoder group not found!")
        except:
            pass

        # Learning rate schedulers
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
        
        # [FIX] VM encoder scheduler removed (now uses base grid scheduler with multiplier)


    def update_learning_rate(self, iteration):
        '''
        Learning rate scheduling per step.
        [FIX] Simplified: VM uses base grid LR with multiplier (no separate schedule).
        '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
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
            # [FIX] VM encoder: use base grid LR with multiplier (no manual grad scaling)
            # [ORIG] Separate vm_planes/vm_mlp with different LRs
            elif param_group["name"] == "vm_encoder" and self.hybrid_config and self.hybrid_config.hybrid_enable:
                base_lr = self.grid_scheduler_args(iteration)
                lr = base_lr * self.hybrid_config.vm_lr_multiplier  # 1.5x boost
                param_group['lr'] = lr
            elif param_group["name"] == "proj_hash" and self.hybrid_config and self.hybrid_config.hybrid_enable:
                lr = self.grid_scheduler_args(iteration)  # Same as grid
                param_group['lr'] = lr
            elif param_group["name"] == "feature_scale" and self.hybrid_config and self.hybrid_config.hybrid_enable:
                lr = self.grid_scheduler_args(iteration) * 0.1  # Small LR if trainable
                param_group['lr'] = lr
        
        # [FIX] hash_gain and gate annealing now handled by trainer (not here)
        # [ORIG] Old hash_gain scheduler with gain_start/gain_end removed

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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

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

    def densification_postfix(self, new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask):
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

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
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

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_scaling_base = self._scaling_base[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_sh_mask = self._sh_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, opt=None, iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # [UNIVERSAL ENHANCEMENT] Adaptive densification using universal parameters
        # These parameters help all scenes with fine details and prevent over-smoothing
        if opt:
            # Use universal adaptive factors instead of dataset-specific modes
            densify_grad_threshold = max_grad * opt.densify_grad_threshold_factor  # 70% of default
            densify_until_iter = int(opt.iterations * opt.densify_until_iter_factor)  # 150% of iterations
            size_threshold = max_screen_size * opt.size_threshold_multiplier if max_screen_size else 0.1 * extent
            final_opacity_thr = opt.prune_opacity_threshold_final  # 0.01 (stricter)
        else:
            # Fallback to original parameters if opt not provided
            densify_grad_threshold = max_grad
            densify_until_iter = float('inf')
            size_threshold = max_screen_size
            final_opacity_thr = min_opacity

        # Only densify before extended densification window
        if iteration is None or iteration < densify_until_iter:
            self.update_attributes()
            self.densify_and_clone(grads, densify_grad_threshold, extent)
            self.update_attributes()
            self.densify_and_split(grads, densify_grad_threshold, extent)

        self.update_attributes()
        
        # Pruning logic with adaptive opacity threshold
        prune_mask = torch.logical_or((self.get_mask <= 0.01).squeeze(), (self.get_opacity < final_opacity_thr).squeeze())
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        # 重新同步派生属性，避免 _scaling 等张量尺寸滞后
        self.update_attributes()

        torch.cuda.empty_cache()

    def mask_prune(self):
        prune_mask = (self.get_mask <= 0.01).squeeze()
        self.prune_points(prune_mask)
        # Pruning 后立即刷新派生属性，保持张量长度一致
        self.update_attributes()
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
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
        
        # [HYBRID] Handle both original HashGrid and HybridFeatureEncoder
        if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
            # For hybrid encoder, extract hash params from wrapped encoder
            grid_params = self._grid.hash.state_dict()["params"].detach().cpu().numpy()
            
            # Save VM encoder parameters (planes, MLP, projections, feature_scale)
            vm_xy = self._grid.vm.xy.detach().cpu().numpy()
            vm_xz = self._grid.vm.xz.detach().cpu().numpy()
            vm_yz = self._grid.vm.yz.detach().cpu().numpy()
            vm_mlp = self._grid.vm.mlp.weight.detach().cpu().numpy()
            proj_hash = self._grid.proj_hash.weight.detach().cpu().numpy()
            feature_scale = self._grid.feature_scale.detach().cpu().item()
            
            np.savez_compressed(os.path.join(path, "vm_planes.npz"), 
                              xy=vm_xy, xz=vm_xz, yz=vm_yz)
            np.savez_compressed(os.path.join(path, "vm_mlp.npz"), data=vm_mlp)
            np.savez_compressed(os.path.join(path, "proj_hash.npz"), data=proj_hash)
            np.savez_compressed(os.path.join(path, "feature_scale.npz"), data=np.array([feature_scale]))
        else:
            # Original path: direct grid params
            grid_params = self._grid.state_dict()["params"].detach().cpu().numpy()

        # G-PCC encoding
        encode_xyz(xyz, path)

        # uniform quantization
        f_dc_quant, f_dc_scale, f_dc_min = quantize(f_dc, bit=8)
        scaling_base_quant, scaling_base_scale, scaling_base_min = quantize(scale_base, bit=8)
        grid_quant, grid_scale, grid_min = quantize(grid_params, bit=6, log=True)

        np.savez_compressed(os.path.join(path, "f_dc.npz"), data=f_dc_quant.astype(np.uint8), scale=f_dc_scale, min=f_dc_min)
        np.savez_compressed(os.path.join(path, "scale_base.npz"), data=scaling_base_quant.astype(np.uint8), scale=scaling_base_scale, min=scaling_base_min)
        np.savez_compressed(os.path.join(path, "grid.npz"), data=grid_quant.astype(np.uint8), scale=grid_scale, min=grid_min)

        # sh mask
        sh_mask = (sh_mask > 0.01).cumprod(axis=-1).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "sh_mask.npz"), data=sh_mask)

        # mlps
        opacity_params = self._opacity_head.state_dict()['params'].half().cpu().detach().numpy()
        f_rest_params = self._features_rest_head.state_dict()['params'].half().cpu().detach().numpy()
        scale_params = self._scaling_head.state_dict()['params'].half().cpu().detach().numpy()
        rotation_params = self._rotation_head.state_dict()['params'].half().cpu().detach().numpy()

        np.savez_compressed(os.path.join(path, "opacity.npz"), data=opacity_params)
        np.savez_compressed(os.path.join(path, "f_rest.npz"), data=f_rest_params)
        np.savez_compressed(os.path.join(path, "scale.npz"), data=scale_params)
        np.savez_compressed(os.path.join(path, "rotation.npz"), data=rotation_params)

        size = 0
        size += os.path.getsize(os.path.join(path, "xyz.bin"))
        print("xyz.bin", os.path.getsize(os.path.join(path, "xyz.bin")) / 1e6)
        size += os.path.getsize(os.path.join(path, "f_dc.npz"))
        print("f_dc.npz", os.path.getsize(os.path.join(path, "f_dc.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "scale_base.npz"))
        print("scale_base.npz", os.path.getsize(os.path.join(path, "scale_base.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "grid.npz"))
        print("grid.npz", os.path.getsize(os.path.join(path, "grid.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "sh_mask.npz"))
        print("sh_mask.npz", os.path.getsize(os.path.join(path, "sh_mask.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "opacity.npz"))
        print("opacity.npz", os.path.getsize(os.path.join(path, "opacity.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "f_rest.npz"))
        print("f_rest.npz", os.path.getsize(os.path.join(path, "f_rest.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "scale.npz"))
        print("scale.npz", os.path.getsize(os.path.join(path, "scale.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "rotation.npz"))
        print("rotation.npz", os.path.getsize(os.path.join(path, "rotation.npz")) / 1e6)
        
        # [HYBRID] VM encoder files if present
        if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
            if os.path.exists(os.path.join(path, "vm_planes.npz")):
                size += os.path.getsize(os.path.join(path, "vm_planes.npz"))
                print("vm_planes.npz", os.path.getsize(os.path.join(path, "vm_planes.npz")) / 1e6)
            if os.path.exists(os.path.join(path, "vm_mlp.npz")):
                size += os.path.getsize(os.path.join(path, "vm_mlp.npz"))
                print("vm_mlp.npz", os.path.getsize(os.path.join(path, "vm_mlp.npz")) / 1e6)
            if os.path.exists(os.path.join(path, "proj_hash.npz")):
                size += os.path.getsize(os.path.join(path, "proj_hash.npz"))
                print("proj_hash.npz", os.path.getsize(os.path.join(path, "proj_hash.npz")) / 1e6)
            if os.path.exists(os.path.join(path, "feature_scale.npz")):
                size += os.path.getsize(os.path.join(path, "feature_scale.npz"))
                print("feature_scale.npz", os.path.getsize(os.path.join(path, "feature_scale.npz")) / 1e6)

        print(f"Total size: {(size / 1e6)} MB")

    def decompress_attributes(self, path):
        # G-PCC decoding
        xyz = decode_xyz(path)
        self._xyz = torch.from_numpy(np.asarray(xyz)).cuda()

        # uniform dequantization
        f_dc_dict = np.load(os.path.join(path, 'f_dc.npz'))
        scale_base_dict = np.load(os.path.join(path, 'scale_base.npz'))
        grid_dict = np.load(os.path.join(path, 'grid.npz'))

        f_dc = dequantize(f_dc_dict['data'], f_dc_dict['scale'], f_dc_dict['min'])
        scaling_base = dequantize(scale_base_dict['data'], scale_base_dict['scale'], scale_base_dict['min'])
        grid_params = dequantize(grid_dict['data'], grid_dict['scale'], grid_dict['min'], log=True)

        self._features_dc = torch.from_numpy(np.asarray(f_dc)).cuda()
        self._scaling_base = torch.from_numpy(np.asarray(scaling_base)).cuda()
        
        # [HYBRID] Handle both original HashGrid and HybridFeatureEncoder
        if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
            # For hybrid encoder, load into wrapped hash encoder
            self._grid.hash.params = nn.Parameter(torch.from_numpy(np.asarray(grid_params)).half().cuda())
            
            # Load VM encoder parameters if they exist
            if os.path.exists(os.path.join(path, "vm_planes.npz")):
                vm_planes = np.load(os.path.join(path, "vm_planes.npz"))
                self._grid.vm.xy.data = torch.from_numpy(vm_planes['xy']).cuda()
                self._grid.vm.xz.data = torch.from_numpy(vm_planes['xz']).cuda()
                self._grid.vm.yz.data = torch.from_numpy(vm_planes['yz']).cuda()
            if os.path.exists(os.path.join(path, "vm_mlp.npz")):
                vm_mlp = np.load(os.path.join(path, "vm_mlp.npz"))['data']
                self._grid.vm.mlp.weight.data = torch.from_numpy(vm_mlp).cuda()
            if os.path.exists(os.path.join(path, "proj_hash.npz")):
                proj_hash = np.load(os.path.join(path, "proj_hash.npz"))['data']
                self._grid.proj_hash.weight.data = torch.from_numpy(proj_hash).cuda()
            if os.path.exists(os.path.join(path, "feature_scale.npz")):
                feature_scale = np.load(os.path.join(path, "feature_scale.npz"))['data'][0]
                self._grid.feature_scale.data = torch.tensor(feature_scale).cuda()
        else:
            # Original path: direct grid params
            self._grid.params = nn.Parameter(torch.from_numpy(np.asarray(grid_params)).half().cuda())
        
        # Recover params for hybrid encoder
        if self.hybrid_config and self.hybrid_config.hybrid_enable and isinstance(self._grid, HybridFeatureEncoder):
             try:
                 self._grid.set_hash_gain(1.0)
                 self._grid.set_gate_sched(self.hybrid_config.gate_alpha_end, self.hybrid_config.gate_tau_end)
                 self._grid.set_beta_min(self.hybrid_config.beta_min_end)
             except Exception:
                 pass

        # sh_mask
        sh_mask = np.load(os.path.join(path, 'sh_mask.npz'))['data'].astype(np.float32)
        self._sh_mask = torch.from_numpy(np.asarray(sh_mask)).cuda()

        # mlps
        opacity_params = np.load(os.path.join(path, 'opacity.npz'))['data']
        f_rest_params = np.load(os.path.join(path, 'f_rest.npz'))['data']
        scale_params = np.load(os.path.join(path, 'scale.npz'))['data']
        rotation_params = np.load(os.path.join(path, 'rotation.npz'))['data']

        self._opacity_head.params = nn.Parameter(torch.from_numpy(np.asarray(opacity_params)).half().cuda())
        self._features_rest_head.params = nn.Parameter(torch.from_numpy(np.asarray(f_rest_params)).half().cuda())
        self._scaling_head.params = nn.Parameter(torch.from_numpy(np.asarray(scale_params)).half().cuda())
        self._rotation_head.params = nn.Parameter(torch.from_numpy(np.asarray(rotation_params)).half().cuda())

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