"""
Geometry-Appearance Dual-Branch Encoder

Implements feature role specialization: Hash branch for appearance (high-frequency
textures), GeoEncoder branch for geometry (low-frequency structure). This design enables
explicit disentanglement of geometric and appearance representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union
from .geo_encoder import GeoEncoder


class GeometryAppearanceEncoder(nn.Module):
    """
    Hybrid Encoder for Gaussians (GlowGS innovation #1).
    
    Design Philosophy:
    - Combines hash grid (local, high-frequency, view-dependent details) with
      tri-plane GeoEncoder (global, low-frequency, smooth structure).
    - Explicit feature role split: geometry_latent for scale/rotation/opacity,
      appearance_latent for SH/color, enabling disentangled learning.
    - Shared latent provides unified local context, then specialized adapters
      add small-dimension residuals for role-specific refinement.
    
    Architecture:
      Input 3D coords → Hash grid + Tri-plane GeoEncoder
                     → Shared projector (C_shared)
                     → Geometry adapter (C_role) / Appearance adapter (C_role)
                     → geometry_latent / appearance_latent
    
    Args:
        hash_encoder: Pre-initialized hash grid encoder with forward() and n_output_dims.
        out_dim (int): Output feature dimension per branch (C_shared).
        geo_channels (int): GeoEncoder output channels.
        geo_resolution (int): Tri-plane spatial resolution.
        geo_rank (int): Low-rank factorization rank.
        feature_role_split (bool): Enable geometry/appearance disentanglement.
    """
    
    def __init__(
        self,
        hash_encoder,
        out_dim: int,
        geo_channels: int = 8,
        geo_resolution: int = 64,
        geo_rank: int = 8,
        feature_role_split: bool = True
    ):
        super().__init__()
        
        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims
        
        self.geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.01
        )
        self.geo_dim = geo_channels
        
        # Feature Modulation (FiLM) Generators
        # Instead of concatenating large features, we use lightweight VM features 
        # to modulate the high-frequency Hash Grid features.
        # Input: geo_dim (VM context) -> Output: hash_dim * 2 (Scale + Shift)
        self.geometry_modulator = nn.Linear(self.geo_dim, self.hash_dim * 2, bias=True)
        self.appearance_modulator = nn.Linear(self.geo_dim, self.hash_dim * 2, bias=True)

        # Fallback fusion layer for non-split mode (preserve original behavior)
        self.fusion_layer = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)

        # Initialize modulators for identity mapping (scale=0, shift=0)
        # This ensures training starts with pure Hash Grid behavior, gradually introducing VM influence
        for layer in [self.geometry_modulator, self.appearance_modulator]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.normal_(self.fusion_layer.weight, std=0.01)
        nn.init.zeros_(self.fusion_layer.bias)
        
        self._out_dim = out_dim
        self.feature_role_split = feature_role_split
        
        # Try to enable torch.compile for the heavy modulation part
        # This uses Triton backend on supported platforms for fusion
        try:
            self._forward_split_compiled = torch.compile(self._forward_split_impl)
        except Exception:
            self._forward_split_compiled = self._forward_split_impl
    
    @property
    def n_output_dims(self) -> int:
        """Output dimension per branch (compatible with tcnn interface)."""
        return self._out_dim
    
    @property
    def role_dim(self) -> int:
        """Dimension of role-specific residual (C_role)."""
        return 0  # No extra dimensions added in FiLM mode
    
    @property
    def geometry_dim(self) -> int:
        """Output dimension for geometry_latent when feature_role_split=True."""
        return self._out_dim
    
    @property
    def appearance_dim(self) -> int:
        """Output dimension for appearance_latent when feature_role_split=True."""
        return self._out_dim
    
    def _forward_split_impl(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        """
        Implementation of FiLM logic, separated for compilation.
        """
        # FiLM (Feature-wise Linear Modulation)
        # VM features (low-freq) modulate Hash features (high-freq)
        
        # 1. Generate modulation parameters from VM features
        g_params = self.geometry_modulator(geo_latent)  # [N, hash_dim * 2]
        a_params = self.appearance_modulator(geo_latent) # [N, hash_dim * 2]
        
        g_scale, g_shift = g_params.chunk(2, dim=-1)
        a_scale, a_shift = a_params.chunk(2, dim=-1)
        
        # 2. Apply modulation: feat = hash * (1 + scale) + shift
        # Using (1 + scale) allows scale to be initialized to 0 for identity
        geometry_latent = hash_latent * (1.0 + g_scale) + g_shift
        appearance_latent = hash_latent * (1.0 + a_scale) + a_shift
        
        # Shared latent is just the raw hash features (or could be averaged)
        shared_latent = hash_latent

        # Clamp for stability
        geometry_latent = torch.clamp(geometry_latent, -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)
        shared_latent = torch.clamp(shared_latent, -10.0, 10.0)

        return shared_latent, geometry_latent, appearance_latent

    def _forward_split(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        """
        Internal forward logic for feature_role_split mode.
        """
        return self._forward_split_compiled(hash_latent, geo_latent)
    
    def forward(
        self, 
        coordinates: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional feature role specialization.
        
        Args:
            coordinates: [N, 3] normalized positions in [-1, 1].
        
        Returns:
            If feature_role_split=True:
                (shared_latent, geometry_latent, appearance_latent)
                - shared_latent: [N, C_shared] unified local code
                - geometry_latent: [N, C_shared + C_role] for scale/rotation/opacity
                - appearance_latent: [N, C_shared + C_role] for SH/color
            If feature_role_split=False:
                fused_latent: [N, out_dim] single feature vector
        """
        # Hash branch: high-frequency for appearance
        # Note: tinycudann outputs float16, need to convert to float32
        hash_latent = self.hash_encoder(coordinates).float()

        # Geo branch: low-frequency for geometry
        geo_latent = self.geo_encoder(coordinates)

        if self.feature_role_split:
            # Torch.compile introduces graph shape cache issues during densification; run eager instead.
            return self._forward_split(hash_latent, geo_latent)
        else:
            # Fallback: fuse both branches into single latent (legacy behaviour)
            combined = torch.cat([hash_latent, geo_latent], dim=1)
            fused_latent = self.fusion_layer(combined)
            return self._stabilize(fused_latent)
    
    def _stabilize(self, x: torch.Tensor) -> torch.Tensor:
        """Numerical stability: fast clamp (NaN/Inf checks are expensive sync operations)."""
        # Fast path: just clamp (NaN/Inf checks cause GPU-CPU sync, very slow)
        return torch.clamp(x, min=-10.0, max=10.0)
