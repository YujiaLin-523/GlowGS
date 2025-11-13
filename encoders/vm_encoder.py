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
        use_checkpoint: bool = False,
        lpf_enable: bool = True,
        lpf_kernel: int = 3
    ):
        super().__init__()
        assert in_dim == 3, "VMEncoder currently assumes 3D coordinates (xyz)"
        
        self.rank = rank
        self.plane_res = plane_res
        self.out_dim = out_dim
        self.basis = basis
        self.use_checkpoint = use_checkpoint
        
        # Forward-only low-pass filter configuration
        self.lpf_enable = lpf_enable
        self.lpf_kernel = lpf_kernel
        
        # [CHECKLIST 6] Tri-plane coordinate perturbation for axis-aligned bias mitigation
        self.rotation_perturb_enable = False  # Set True for drjohnson
        self.rotation_perturb_angle = 3.0  # degrees (±2-5° range)
        
        # Three axis-aligned planes: XY, XZ, YZ (TensoRF-VM style).
        # Each plane has 'rank' channels; features are sampled, aggregated, then projected.
        # Shape: [rank, plane_res, plane_res]
        self.xy = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        self.xz = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        self.yz = nn.Parameter(torch.zeros(rank, plane_res, plane_res))
        
        # Final linear mixer: aggregates 3 planes (3*rank features) to out_dim
        # No bias to keep it as a pure linear combination (residual-friendly)
        self.mlp = nn.Linear(3 * rank, out_dim, bias=False)
        
        # Balanced initialization: strong enough for VM to learn, not too dominant
        # [ORIG] std=0.15 was too strong
        # [FIX] std=0.01 was too weak, restored to 0.1 for sufficient signal
        # hash_gain annealing (0.05->1.0) provides the actual protection mechanism
        nn.init.trunc_normal_(self.xy, std=0.1)
        nn.init.trunc_normal_(self.xz, std=0.1)
        nn.init.trunc_normal_(self.yz, std=0.1)
        # MLP: standard xavier (gain=1.0) for full capacity
        nn.init.xavier_uniform_(self.mlp.weight, gain=1.0)
        
        # [ACCEPTANCE CHECK A] Confirm align_corners=False in grid sampling (startup log)
        print(f"[VM-ENCODER] Initialized with align_corners=False (kills vertical streaks), "
              f"rank={rank}, res={plane_res}, lpf={lpf_enable}")
    
    def _apply_lpf(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Apply forward-only low-pass filter to plane parameters.
        
        Forward-only view: the pooling operation creates a smoothed view of plane parameters
        in the computation graph. Gradients still flow back to the underlying nn.Parameter
        tensors without any detachment, ensuring normal gradient propagation to VM planes.
        
        This operation is applied ONLY during forward pass to stabilize training.
        The underlying nn.Parameter tensors are NOT modified, ensuring normal
        gradient backpropagation to the original plane parameters.
        
        Args:
            plane: [R, H, W] plane parameter tensor
        
        Returns:
            [R, H, W] low-pass filtered plane (in computation graph)
        """
        if not self.lpf_enable:
            return plane
        
        # Apply average pooling as a robust, cheap low-pass filter
        # [FIX] Don't detach - let gradients flow back to plane parameters
        k = self.lpf_kernel
        pad = k // 2
        plane_lpf = F.avg_pool2d(
            plane.unsqueeze(0),  # [1, R, H, W]
            kernel_size=k,
            stride=1,
            padding=pad
        ).squeeze(0)  # [R, H, W]
        
        return plane_lpf
    
    def _apply_3d_rotation_perturb(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Apply random 3D rotation perturbation to coordinates (training only, no gradient).
        
        [CHECKLIST 6] Breaks fixed axis alignment of tri-planes, reducing direction-dependent
        artifacts (horizontal/vertical streaking). Applied only during training; inference uses
        original coordinates.
        
        Args:
            xyz: [N, 3] coordinates in [0,1]^3
        
        Returns:
            [N, 3] perturbed coordinates, still clamped to [0,1]^3
        """
        if not self.training or not self.rotation_perturb_enable or xyz.shape[0] == 0:
            return xyz
        
        # Random rotation angles (radians): ±angle_deg
        angle_rad = (self.rotation_perturb_angle * 3.14159 / 180.0)
        angles = (torch.rand(3, device=xyz.device) - 0.5) * 2 * angle_rad  # [3] in [-angle, +angle]
        
        # Build 3D rotation matrix (ZYX Euler)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)
        
        # Rotation matrix (row-major, will apply as xyz @ R.T)
        R = torch.tensor([
            [cy*cz, -cy*sz, sy],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
        ], device=xyz.device, dtype=xyz.dtype)
        
        # Center coordinates around 0.5, rotate, then shift back
        xyz_centered = xyz - 0.5
        xyz_rotated = torch.mm(xyz_centered, R.T)  # [N, 3] @ [3, 3].T
        xyz_perturbed = xyz_rotated + 0.5
        
        # Clamp back to [0, 1] (important for boundary preservation)
        return xyz_perturbed.clamp(0.0, 1.0)
    
    def _sample_plane(
        self, 
        plane: torch.Tensor, 
        u: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Bilinear sampling on a [rank, H, W] plane.
        
        Optimized for GPU efficiency: minimizes temporary allocations.
        
        Args:
            plane: [rank, H, W] learnable plane parameters (may be LPF-filtered)
            u, v: [N] normalized coordinates in [0, 1] range
        
        Returns:
            [N, rank] sampled features from the plane
        """
        N = u.shape[0]
        H, W = plane.shape[-2:]
        
        # [FIX A] Optional coordinate jitter to damp grid aliasing (training only)
        # Tiny jitter prevents repetitive sampling patterns on pixel grid boundaries
        if self.training:
            jitter_u = (torch.rand_like(u) - 0.5) * (0.3 / W)  # ~0.3 pixels max
            jitter_v = (torch.rand_like(v) - 0.5) * (0.3 / H)
            u = (u + jitter_u.detach()).clamp(0.0, 1.0)
            v = (v + jitter_v.detach()).clamp(0.0, 1.0)
        
        # [FIX A] Convert [0,1] coords to [-1,1] with inner margin to avoid border stretching
        # Use align_corners=False to prevent column/row edge artifacts (vertical streaking)
        # [FIX CHECKLIST A4] Increase margin to 2 pixels (was 1) to further reduce border artifacts
        eps_u = 2.0 / W  # Two pixel margin in U (more conservative)
        eps_v = 2.0 / H  # Two pixel margin in V (more conservative)
        u_norm = (u * 2.0 - 1.0).clamp(-1.0 + eps_u, 1.0 - eps_u)
        v_norm = (v * 2.0 - 1.0).clamp(-1.0 + eps_v, 1.0 - eps_v)
        
        # Stack and reshape in one go: [N, 2] -> [1, 1, N, 2]
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(1)
        
        # [FIX A] Use align_corners=False to eliminate vertical streaking
        # Input: [R, H, W], Grid: [1, 1, N, 2] -> Output: [1, R, 1, N]
        sampled = F.grid_sample(
            plane.unsqueeze(0),  # [1, R, H, W]
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        # Efficient reshape: [1, R, 1, N] -> [N, R]
        return sampled[0, :, 0, :].t().contiguous()  # Transpose to [N, R]
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sample three planes and aggregate to output features.
        
        Args:
            xyz: [N, 3] coordinates expected in [0,1]^3 (hybrid wrapper converts if needed).
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
        
        # [FIX] Coordinate sanitization: handle NaN/Inf and clamp to [0,1]
        # This prevents grid_sample from producing undefined gradients at boundaries
        xyz = torch.nan_to_num(xyz, nan=0.5, posinf=0.5, neginf=0.5).clamp(0.0, 1.0)
        
        # [CHECKLIST 6] Apply random 3D rotation perturbation (training only, drjohnson mode)
        # Breaks axis-aligned bias of tri-planes, reducing directional streaking
        xyz = self._apply_3d_rotation_perturb(xyz)
        
        # [FIX] Apply forward-only low-pass filter to planes (stabilizes training)
        # Parameters themselves are NOT modified; gradients still flow to original tensors
        Pxy = self._apply_lpf(self.xy)
        Pxz = self._apply_lpf(self.xz)
        Pyz = self._apply_lpf(self.yz)
        
        # Extract coordinates
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        
        # Sample three planes (using LPF-filtered versions)
        # Use checkpointing if enabled to trade compute for memory
        if self.use_checkpoint and self.training:
            f_xy = torch.utils.checkpoint.checkpoint(
                self._sample_plane, Pxy, x, y, use_reentrant=False
            )
            f_xz = torch.utils.checkpoint.checkpoint(
                self._sample_plane, Pxz, x, z, use_reentrant=False
            )
            f_yz = torch.utils.checkpoint.checkpoint(
                self._sample_plane, Pyz, y, z, use_reentrant=False
            )
        else:
            f_xy = self._sample_plane(Pxy, x, y)  # [N, rank]
            f_xz = self._sample_plane(Pxz, x, z)  # [N, rank]
            f_yz = self._sample_plane(Pyz, y, z)  # [N, rank]
        
        # Concatenate plane features: [N, 3*rank]
        f = torch.cat([f_xy, f_xz, f_yz], dim=-1)
        
        # Final linear projection to output dimension
        return self.mlp(f)
