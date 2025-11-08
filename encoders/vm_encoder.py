#
# VM (Vector-Matrix) Encoder for LocoGS Hybrid Feature Encoding
# Low-rank, tri-plane style encoder for capturing low-frequency global structure
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class VMEncoder(nn.Module):
    """
    Vector-Matrix (VM) low-rank encoder for 3D coordinates.
    
    Intended to capture low-frequency, global structure with small parameter count.
    Uses three axis-aligned planes (XY, XZ, YZ) with low-rank factorization,
    inspired by TensoRF-VM decomposition.
    
    Efficient: computes per-axis projections via bilinear sampling and combines 
    with lightweight matmuls.
    
    Args:
        in_dim (int): Input coordinate dimension (must be 3 for xyz).
        rank (int): Low-rank size for each plane (e.g., 8-16). 
                    Controls capacity vs memory tradeoff.
        plane_res (int): Per-plane spatial resolution (e.g., 128-256).
                         Higher resolution captures finer low-frequency details.
        out_dim (int): Output feature dimension after final MLP projection.
        basis (str): Interpolation basis for plane sampling. 
                     Currently supports 'bilinear' (default).
        use_checkpoint (bool): Enable gradient checkpointing for memory saving.
                               Trades compute for memory; useful for very large scenes.
    
    Returns:
        vm_feat: [N, out_dim] feature tensor suitable for fusion with HashGrid.
    
    Invariants:
        - Input xyz must be normalized to [0,1]^3 range.
        - Output dimension matches downstream feature consumer expectations.
        - Small initialization ensures near-zero residual at training start.
    """
    
    def __init__(
        self, 
        in_dim: int = 3, 
        rank: int = 12, 
        plane_res: int = 192,
        out_dim: int = 32, 
        basis: str = 'bilinear', 
        use_checkpoint: bool = False
    ):
        super().__init__()
        assert in_dim == 3, "VMEncoder currently assumes 3D coordinates (xyz)"
        
        self.rank = rank
        self.plane_res = plane_res
        self.out_dim = out_dim
        self.basis = basis
        self.use_checkpoint = use_checkpoint
        
        # Three axis-aligned planes: XY, XZ, YZ (TensoRF-VM style).
        # Each plane has 'rank' channels; features are sampled, aggregated, then projected.
        # Shape: [rank, plane_res, plane_res]
        self.xy = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        self.xz = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        self.yz = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        
        # Final linear mixer: aggregates 3 planes (3*rank features) to out_dim
        # No bias to keep it as a pure linear combination (residual-friendly)
        self.mlp = nn.Linear(3 * rank, out_dim, bias=False)
        
        # [FIX] Initialization: increase std for stronger initial gradients
        # std=0.3 provides sufficient signal without overwhelming Hash encoder
        nn.init.trunc_normal_(self.xy, std=0.3)
        nn.init.trunc_normal_(self.xz, std=0.3)
        nn.init.trunc_normal_(self.yz, std=0.3)
        # [FIX] MLP with higher gain to amplify plane features
        nn.init.xavier_uniform_(self.mlp.weight, gain=1.0)
    
    def _sample_plane(
        self, 
        plane: torch.Tensor, 
        u: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Bilinear sampling on a [rank, H, W] plane.
        
        Uses F.grid_sample for efficient vectorized interpolation.
        Coordinates are clamped to plane boundaries (border padding mode).
        
        Args:
            plane: [rank, H, W] learnable plane parameters
            u, v: [N] normalized coordinates in [0, 1] range
        
        Returns:
            [N, rank] sampled features from the plane
        """
        N = u.shape[0]
        R, H, W = plane.shape
        
        # Convert [0,1] coords to [-1,1] range expected by grid_sample
        # Reshape to [1, H', W', 2] where H'=1, W'=N (batch of N points)
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(1, 1, N, 2)
        
        # Add batch dimension to plane: [1, R, H, W]
        plane_b = plane.unsqueeze(0)
        
        # Sample with align_corners=True for stable interpolation semantics
        # Input: [1, R, H, W], Grid: [1, 1, N, 2] -> Output: [1, R, 1, N]
        sampled = F.grid_sample(
            plane_b, 
            grid, 
            mode='bilinear', 
            padding_mode='border',  # clamp to edge for stability
            align_corners=True
        )
        
        # Reshape to [N, R]: squeeze spatial dims and transpose
        sampled = sampled.squeeze(2).squeeze(0).transpose(0, 1)  # [1,R,1,N] -> [R,N] -> [N,R]
        return sampled
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sample three planes and aggregate to output features.
        
        Args:
            xyz: [N, 3] coordinates in normalized [0,1]^3 space.
                 Typically obtained from Gaussian centers after contraction.
        
        Returns:
            [N, out_dim] VM features ready for fusion with HashGrid.
        
        Math:
            For each point (x,y,z):
            - Sample XY plane at (x,y) -> [rank] features
            - Sample XZ plane at (x,z) -> [rank] features  
            - Sample YZ plane at (y,z) -> [rank] features
            - Concatenate to [3*rank]
            - Linear projection to [out_dim]
        """
        N = xyz.shape[0]
        
        # [MEMORY FIX] Batch processing to prevent OOM with large point clouds
        # grid_sample creates large intermediate tensors, so we split into chunks
        chunk_size = 16384  # Process 16k points at a time (balanced for 24GB VRAM)
        
        if N > chunk_size and self.training:
            # Chunked processing for large batches
            num_chunks = (N + chunk_size - 1) // chunk_size
            outputs = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, N)
                xyz_chunk = xyz[start_idx:end_idx]
                
                # Extract coordinates for this chunk
                x, y, z = xyz_chunk[..., 0], xyz_chunk[..., 1], xyz_chunk[..., 2]
                
                # Sample three planes
                if self.use_checkpoint:
                    f_xy = torch.utils.checkpoint.checkpoint(
                        self._sample_plane, self.xy, x, y, use_reentrant=False
                    )
                    f_xz = torch.utils.checkpoint.checkpoint(
                        self._sample_plane, self.xz, x, z, use_reentrant=False
                    )
                    f_yz = torch.utils.checkpoint.checkpoint(
                        self._sample_plane, self.yz, y, z, use_reentrant=False
                    )
                else:
                    f_xy = self._sample_plane(self.xy, x, y)
                    f_xz = self._sample_plane(self.xz, x, z)
                    f_yz = self._sample_plane(self.yz, y, z)
                
                # Concatenate and project
                f = torch.cat([f_xy, f_xz, f_yz], dim=-1)
                out_chunk = self.mlp(f)
                outputs.append(out_chunk)
            
            # Concatenate all chunks
            return torch.cat(outputs, dim=0)
        
        else:
            # Original path for small batches or inference
            # Extract coordinates
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            
            # Sample three planes (vectorized for speed)
            # Use checkpointing if enabled to trade compute for memory
            if self.use_checkpoint and self.training:
                f_xy = torch.utils.checkpoint.checkpoint(
                    self._sample_plane, self.xy, x, y, use_reentrant=False
                )
                f_xz = torch.utils.checkpoint.checkpoint(
                    self._sample_plane, self.xz, x, z, use_reentrant=False
                )
                f_yz = torch.utils.checkpoint.checkpoint(
                    self._sample_plane, self.yz, y, z, use_reentrant=False
                )
            else:
                f_xy = self._sample_plane(self.xy, x, y)  # [N, rank]
                f_xz = self._sample_plane(self.xz, x, z)  # [N, rank]
                f_yz = self._sample_plane(self.yz, y, z)  # [N, rank]
            
            # Concatenate plane features: [N, 3*rank]
            f = torch.cat([f_xy, f_xz, f_yz], dim=-1)
            
            # Final linear projection to output dimension
            return self.mlp(f)
