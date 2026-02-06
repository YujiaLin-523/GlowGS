"""
GeoEncoder — TensoRF-style Vector-Matrix (VM) Decomposition

Architecture
    Raw world xyz  ──► L∞ contraction (AABB → [-1,1]³) ──►
    ┌─ Plane XY(x,y) × Line Z(z) ─┐
    │  Plane XZ(x,z) × Line Y(y)  │ ──► Element-wise Sum ──► Linear ──► geo_feat [N, C]
    └─ Plane YZ(y,z) × Line X(x) ─┘

Key design choices (matching TensoRF):
  • grid_sample:  align_corners=True, padding_mode='zeros'
  • Lines are *vertical* strips [1, R, Res, 1] sampled with (0, u)
  • Fusion = Hadamard product per branch, then Sum (not Concat)
  • init_scale = 0.1  so  plane×line ≈ 0.01  at start
  • Built-in L∞ (cuboid) contraction for unbounded scenes
  • Progressive resolution upsampling via upsample_resolution()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    """
    Geometry encoder using VM decomposition with built-in L∞ contraction.

    Args:
        resolution:   Spatial resolution of planes / lines (default 128).
        rank:         Number of VM components per branch (default 48).
        out_channels: Output feature dimension after projection (default 32).
        init_scale:   Std of parameter initialization (default 0.1).
    """

    def __init__(
        self,
        resolution: int = 128,
        rank: int = 48,
        out_channels: int = 32,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.out_channels = out_channels

        # ----- VM Parameters -----
        # Planes (Matrices): [1, R, Res, Res]
        self.plane_xy = nn.Parameter(torch.randn(1, rank, resolution, resolution) * init_scale)
        self.plane_xz = nn.Parameter(torch.randn(1, rank, resolution, resolution) * init_scale)
        self.plane_yz = nn.Parameter(torch.randn(1, rank, resolution, resolution) * init_scale)

        # Lines (Vectors): [1, R, Res, 1]  — vertical strips
        self.line_z = nn.Parameter(torch.randn(1, rank, resolution, 1) * init_scale)
        self.line_y = nn.Parameter(torch.randn(1, rank, resolution, 1) * init_scale)
        self.line_x = nn.Parameter(torch.randn(1, rank, resolution, 1) * init_scale)

        # ----- Projection -----
        # VM sum yields `rank` channels → project to `out_channels`
        self.projection = nn.Linear(rank, out_channels)
        nn.init.normal_(self.projection.weight, std=init_scale)
        nn.init.zeros_(self.projection.bias)

        # ----- Compilation -----
        try:
            self._vm_forward = torch.compile(self._vm_forward_impl)
        except Exception:
            self._vm_forward = self._vm_forward_impl

    # ------------------------------------------------------------------
    # L∞ (cuboid) contraction
    # ------------------------------------------------------------------
    @staticmethod
    def contract_linf(xyz: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
        """
        Map world coordinates to [-1, 1]³ using L-infinity (cuboid) contraction.

        Inside the AABB → linear mapping to [-1, 1].
        Outside the AABB → compressed into the boundary shell via L∞ norm,
        so that the entire unbounded space maps into [-1, 1]³.

        Args:
            xyz:  [N, 3]  raw world positions.
            aabb: [6]     (min_x, min_y, min_z, max_x, max_y, max_z).
        Returns:
            contracted: [N, 3] in [-1, 1].
        """
        aabb_min = aabb[:3]
        aabb_max = aabb[3:]
        aabb_center = (aabb_min + aabb_max) * 0.5
        aabb_half   = (aabb_max - aabb_min) * 0.5

        # Normalise so that AABB → [-1, 1]
        x = (xyz - aabb_center) / aabb_half.clamp(min=1e-6)

        # L∞ contraction for points outside the unit cube
        linf = x.abs().max(dim=-1, keepdim=True).values  # [N, 1]
        mask = (linf > 1.0).squeeze(-1)                  # [N]
        if mask.any():
            # contract: x_contracted = x / |x|_inf * (2 - 1/|x|_inf)
            # Maps |x|_inf ∈ (1, ∞) → (1, 2), preserving direction
            scale = (2.0 - 1.0 / linf[mask]) / linf[mask]
            x[mask] = x[mask] * scale

        return x.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # grid_sample helpers
    # ------------------------------------------------------------------
    def _sample_plane(self, plane: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Bilinear sample from a 2-D plane.
        plane: [1, R, Res, Res]   grid: [1, 1, N, 2]  → [1, R, 1, N]
        """
        return F.grid_sample(
            plane, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )

    def _sample_line(self, line: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Bilinear sample from a 1-D line (vertical strip).
        line: [1, R, Res, 1]   grid: [1, 1, N, 2]  → [1, R, 1, N]
        """
        return F.grid_sample(
            line, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )

    # ------------------------------------------------------------------
    # VM forward (compilable core)
    # ------------------------------------------------------------------
    def _vm_forward_impl(
        self, coords,
        p_xy, p_xz, p_yz,
        l_z, l_y, l_x,
    ):
        """
        coords: [N, 3] in [-1, 1]   (already contracted)
        """
        N = coords.shape[0]
        _zero = torch.zeros(N, 1, device=coords.device, dtype=coords.dtype)

        # 2-D plane grids
        grid_xy = coords[:, [0, 1]].view(1, 1, N, 2)   # (x, y)
        grid_xz = coords[:, [0, 2]].view(1, 1, N, 2)   # (x, z)
        grid_yz = coords[:, [1, 2]].view(1, 1, N, 2)   # (y, z)

        # 1-D line grids (no in-place ops — safe for torch.compile)
        # Lines are [1, R, Res, 1] (vertical), so sample with (0, u):
        #   grid_sample interprets grid as (w_coord, h_coord)
        #   width=1 → w_coord=0 always;  height=Res → h_coord=u
        grid_z = torch.cat([_zero, coords[:, 2:3]], dim=1).view(1, 1, N, 2)
        grid_y = torch.cat([_zero, coords[:, 1:2]], dim=1).view(1, 1, N, 2)
        grid_x = torch.cat([_zero, coords[:, 0:1]], dim=1).view(1, 1, N, 2)

        vm_xy = self._sample_plane(p_xy, grid_xy) * self._sample_line(l_z, grid_z)
        vm_xz = self._sample_plane(p_xz, grid_xz) * self._sample_line(l_y, grid_y)
        vm_yz = self._sample_plane(p_yz, grid_yz) * self._sample_line(l_x, grid_x)

        # Aggregate via element-wise sum
        vm_sum = (vm_xy + vm_xz + vm_yz)          # [1, R, 1, N]
        vm_feat = vm_sum.squeeze(0).squeeze(1).t() # [N, R]

        # Projection
        return self.projection(vm_feat)            # [N, out_channels]

    # ------------------------------------------------------------------
    # Progressive upsampling
    # ------------------------------------------------------------------
    def upsample_resolution(self, new_resolution: int):
        """
        Upsample all 6 VM parameters (3 planes + 3 lines) to *new_resolution*
        using bilinear interpolation.  Typically called once at step ~7000.
        """
        if new_resolution == self.resolution:
            return
        device = self.plane_xy.device
        dtype  = self.plane_xy.dtype

        for name in ['plane_xy', 'plane_xz', 'plane_yz']:
            old = getattr(self, name)                                        # [1, R, H, W]
            new = F.interpolate(old.data, size=(new_resolution, new_resolution),
                                mode='bilinear', align_corners=True)
            setattr(self, name, nn.Parameter(new.to(device=device, dtype=dtype)))

        for name in ['line_z', 'line_y', 'line_x']:
            old = getattr(self, name)                                        # [1, R, H, 1]
            new = F.interpolate(old.data, size=(new_resolution, 1),
                                mode='bilinear', align_corners=True)
            setattr(self, name, nn.Parameter(new.to(device=device, dtype=dtype)))

        self.resolution = new_resolution
        print(f"[GeoEncoder] Upsampled VM resolution → {new_resolution}")

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, coordinates: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: [N, 3] raw world positions (unbounded).
            aabb:        [6]    scene bounding box.
        Returns:
            geo_feat: [N, out_channels]
        """
        coords = self.contract_linf(coordinates, aabb)

        return self._vm_forward(
            coords,
            self.plane_xy, self.plane_xz, self.plane_yz,
            self.line_z, self.line_y, self.line_x,
        )
