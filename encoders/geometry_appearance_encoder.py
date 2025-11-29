"""
Geometry-Appearance Dual-Branch Encoder

Implements feature role specialization: Hash branch for appearance (high-frequency
textures), GeoEncoder branch for geometry (low-frequency structure). This design enables
explicit disentanglement of geometric and appearance representations.

Architecture:
    Input: 3D normalized coordinates
    ↓
    ┌───────────────────┬─────────────────────┐
    │   Hash Branch     │   GeoEncoder Branch │
    │  (Appearance)     │  (Geometry)         │
    │  High-freq detail │  Low-freq structure │
    └─────────┬─────────┴──────────┬──────────┘
              ↓                    ↓
        appearance_latent    geometry_latent
              ↓                    ↓
          SH/Color heads     Scale/Rot/Opacity heads

Author: GlowGS Project
License: See LICENSE.md
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
        
        # Projection layers for each branch
        self.appearance_projection = nn.Linear(self.hash_dim, out_dim, bias=True)
        self.geometry_projection = nn.Linear(self.geo_dim, out_dim, bias=True)
        
        # Fallback fusion layer for non-split mode
        self.fusion_layer = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)
        
        # Initialize with small weights for stability
        for layer in [self.appearance_projection, self.geometry_projection, self.fusion_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        
        self._out_dim = out_dim
        self.feature_role_split = feature_role_split
    
    @property
    def n_output_dims(self) -> int:
        """Output dimension per branch (compatible with tcnn interface)."""
        return self._out_dim
    
    def forward(
        self, 
        coordinates: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional feature role specialization.
        
        Args:
            coordinates: [N, 3] normalized positions in [-1, 1].
        
        Returns:
            If feature_role_split=True:
                (geometry_latent, appearance_latent): Both [N, out_dim]
            If feature_role_split=False:
                fused_latent: [N, out_dim] single feature vector
        """
        # Hash branch: high-frequency for appearance
        # Note: tinycudann outputs float16, need to convert to float32
        hash_latent = self.hash_encoder(coordinates).float()
        
        # Geo branch: low-frequency for geometry
        geo_latent = self.geo_encoder(coordinates)
        
        if self.feature_role_split:
            # Geometry latent from GeoEncoder (structural/low-frequency bias)
            geometry_latent = self.geometry_projection(geo_latent)
            geometry_latent = self._stabilize(geometry_latent)
            
            # Appearance latent from Hash (detailed/high-frequency bias)
            appearance_latent = self.appearance_projection(hash_latent)
            appearance_latent = self._stabilize(appearance_latent)
            
            return geometry_latent, appearance_latent
        else:
            # Fallback: fuse both branches into single latent
            combined = torch.cat([hash_latent, geo_latent], dim=1)
            fused_latent = self.fusion_layer(combined)
            return self._stabilize(fused_latent)
    
    def _stabilize(self, x: torch.Tensor) -> torch.Tensor:
        """Numerical stability: clamp extreme values."""
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.clamp(x, min=-1e6, max=1e6)
