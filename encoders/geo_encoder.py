"""
Geometry Encoder (GeoEncoder)

A lightweight tri-plane based encoder that captures global low-frequency 
geometric structure to complement the local high-frequency details from 
hash grid encoding.

Key Features:
- Tri-plane factorization (XY, XZ, YZ planes) for memory efficiency
- Low-rank decomposition for each plane to reduce parameters
- Smooth bilinear interpolation for continuous feature queries
- Small initialization to serve as a gentle structural prior

Architecture:
    Input: 3D coordinates (x, y, z) ∈ [-1, 1]³
    ↓
    Sample from 3 orthogonal planes (XY, XZ, YZ)
    ↓
    Concatenate plane features
    ↓
    Linear projection to output dimension
    ↓
    Output: Geometric structure feature vector

Author: GlowGS Project
License: See LICENSE.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    """
    Geometry Encoder using tri-plane factorization.
    
    Captures smooth, global geometric features across 3D space by representing
    the feature field as three orthogonal 2D planes. Each plane uses low-rank 
    decomposition for parameter efficiency.
    
    Args:
        resolution (int): Spatial resolution of each plane (e.g., 64 for 64x64).
        rank (int): Rank for low-rank factorization of plane features.
        out_channels (int): Number of output feature channels.
        init_scale (float): Initialization scale for plane parameters.
                           Small values ensure GeoEncoder acts as gentle prior.
    
    Attributes:
        plane_xy (nn.Parameter): XY plane features [resolution, resolution, rank]
        plane_xz (nn.Parameter): XZ plane features [resolution, resolution, rank]
        plane_yz (nn.Parameter): YZ plane features [resolution, resolution, rank]
        projection (nn.Linear): Final linear layer to project concatenated 
                               plane features to output dimension.
    
    Example:
        >>> encoder = GeoEncoder(resolution=64, rank=8, out_channels=8)
        >>> coords = torch.randn(1000, 3).cuda()  # [N, 3] in range [-1, 1]
        >>> features = encoder(coords)  # [N, 8]
    """
    
    def __init__(
        self,
        resolution: int = 64,
        rank: int = 8,
        out_channels: int = 8,
        init_scale: float = 0.01
    ):
        super().__init__()
        
        self.resolution = resolution
        self.rank = rank
        self.out_channels = out_channels
        
        # Initialize tri-plane parameters with small random values
        # Shape: [resolution, resolution, rank] for each plane
        self.plane_xy = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_xz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_yz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        
        # Project concatenated tri-plane features to output dimension
        # Input: 3 * rank (from 3 planes), Output: out_channels
        self.projection = nn.Linear(3 * rank, out_channels, bias=True)
        
        # Initialize projection weights small to maintain gentle contribution
        nn.init.normal_(self.projection.weight, std=init_scale)
        nn.init.zeros_(self.projection.bias)
    
    def _sample_plane(
        self,
        plane: torch.Tensor,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample features from a 2D plane using bilinear interpolation.
        Optimized for minimal memory allocation.
        
        Args:
            plane (torch.Tensor): Plane parameters [H, W, rank]
            coordinates (torch.Tensor): 2D coordinates [N, 2] in range [-1, 1]
        
        Returns:
            torch.Tensor: Sampled features [N, rank]
        """
        # Use contiguous() only if needed, and avoid unnecessary reshape operations
        # Reshape plane: [H, W, rank] -> [1, rank, H, W] for grid_sample
        plane_features = plane.permute(2, 0, 1).unsqueeze(0)
        
        # Reshape coordinates: [N, 2] -> [1, 1, N, 2] for grid_sample
        grid = coordinates.view(1, 1, -1, 2)
        
        # Bilinear interpolation - use mode='bilinear' for smooth features
        # Output: [1, rank, 1, N] -> [N, rank]
        sampled = F.grid_sample(
            plane_features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        # Efficient reshape: [1, rank, 1, N] -> [N, rank]
        return sampled.view(self.rank, -1).t()
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute global low-frequency features.
        Optimized for real-time inference with minimal memory allocation.
        
        Args:
            coordinates (torch.Tensor): 3D coordinates [N, 3] in range [-1, 1]
                                       (x, y, z) format
        
        Returns:
            torch.Tensor: Geometry features [N, out_channels]
        """
        # Fast path: clamp in-place if coordinates are not used elsewhere
        # Clamp coordinates to [-1, 1] to prevent illegal memory access
        coords = coordinates.clamp(-1.0, 1.0)
        
        # Efficient tri-plane sampling using slicing instead of separate cat operations
        # XY plane: sample with (x, y)
        xy_features = self._sample_plane(self.plane_xy, coords[:, :2])
        # XZ plane: sample with (x, z)  
        xz_features = self._sample_plane(self.plane_xz, coords[:, [0, 2]])
        # YZ plane: sample with (y, z)
        yz_features = self._sample_plane(self.plane_yz, coords[:, 1:])
        
        # Concatenate features from all three planes
        triplane_features = torch.cat([xy_features, xz_features, yz_features], dim=1)
        
        # Project to output dimension with clamping for numerical stability
        output = self.projection(triplane_features)
        
        # Fast clamp for stability
        return output.clamp(-10.0, 10.0)
