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
    Dual-branch encoder with explicit geometry/appearance feature role split.
    
    Hash branch captures high-frequency details optimal for appearance (SH/color).
    GeoEncoder branch provides smooth low-frequency priors optimal for geometry (scale/rot).
    
    Args:
        hash_encoder: Pre-initialized hash grid encoder with forward() and n_output_dims.
        out_dim (int): Output feature dimension per branch.
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
        
        # Shared latent projector: unified local code from dual-branch features
        # shared_latent dimension equals baseline out_dim to preserve downstream compatibility
        self.shared_projector = nn.Sequential(
            nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True),
            nn.SiLU()
        )

        # Small-dimension residual adapters: lightweight role-specific specialization
        # C_role << C_shared (e.g., 16 vs 64) to reduce computation and memory
        self._role_dim = 16  # Fixed small dimension for residual, not exposed as CLI param
        self.geometry_adapter = nn.Linear(out_dim, self._role_dim, bias=True)
        self.appearance_adapter = nn.Linear(out_dim, self._role_dim, bias=True)

        # Fallback fusion layer for non-split mode (preserve original behavior)
        self.fusion_layer = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)

        # Initialize small weights for stability
        for layer in [self.shared_projector[0], self.geometry_adapter, self.appearance_adapter, self.fusion_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        
        self._out_dim = out_dim
        self.feature_role_split = feature_role_split
    
    @property
    def n_output_dims(self) -> int:
        """Output dimension per branch (compatible with tcnn interface)."""
        return self._out_dim
    
    @property
    def role_dim(self) -> int:
        """Dimension of role-specific residual (C_role)."""
        return self._role_dim if self.feature_role_split else 0
    
    @property
    def geometry_dim(self) -> int:
        """Output dimension for geometry_latent when feature_role_split=True."""
        return self._out_dim + self._role_dim if self.feature_role_split else self._out_dim
    
    @property
    def appearance_dim(self) -> int:
        """Output dimension for appearance_latent when feature_role_split=True."""
        return self._out_dim + self._role_dim if self.feature_role_split else self._out_dim
    
    def _forward_split(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        """
        Internal forward logic for feature_role_split mode.
        Separated for torch.compile optimization.
        """
        # Build shared latent from concatenated multi-resolution features
        shared_input = torch.cat([hash_latent, geo_latent], dim=1)
        shared_latent = self.shared_projector(shared_input)

        # Small-dimension residual adapters: lightweight role-specific specialization
        geometry_residual = self.geometry_adapter(shared_latent)   # [N, C_role]
        appearance_residual = self.appearance_adapter(shared_latent)  # [N, C_role]

        # Concatenate shared_latent with role-specific residual
        geometry_latent = torch.cat([shared_latent, geometry_residual], dim=-1)   # [N, C_shared + C_role]
        appearance_latent = torch.cat([shared_latent, appearance_residual], dim=-1)  # [N, C_shared + C_role]

        # Single clamp at the end (avoid multiple clamp operations)
        geometry_latent = torch.clamp(geometry_latent, -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)
        shared_latent = torch.clamp(shared_latent, -10.0, 10.0)

        return shared_latent, geometry_latent, appearance_latent
    
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
