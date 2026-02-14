import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    """
    Geometry encoder using tri-plane factorization.

    Represents the feature field as three orthogonal 2D planes with
    low-rank parameterisation for memory efficiency.

    Args:
        resolution:  Spatial resolution of each plane (e.g. 96 → 96×96).
        rank:        Rank per plane for low-rank factorisation.
        out_channels: Number of output feature channels.
        init_scale:  Initialisation scale for plane parameters.

    Example::

        >>> encoder = GeoEncoder(resolution=96, rank=12, out_channels=8)
        >>> coords = torch.randn(1000, 3).cuda().clamp(-1, 1)
        >>> features = encoder(coords)   # [1000, 8]
    """

    def __init__(
        self,
        resolution: int = 96,
        rank: int = 12,
        out_channels: int = 8,
        init_scale: float = 0.01,
    ):
        super().__init__()

        self.resolution = resolution
        self.rank = rank
        self.out_channels = out_channels

        # Tri-plane parameters: [resolution, resolution, rank]
        self.plane_xy = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_xz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )
        self.plane_yz = nn.Parameter(
            torch.randn(resolution, resolution, rank) * init_scale
        )

        # Projection: 3 * rank → out_channels
        self.projection = nn.Linear(3 * rank, out_channels, bias=True)
        nn.init.normal_(self.projection.weight, std=init_scale)
        nn.init.zeros_(self.projection.bias)

        # Optional torch.compile acceleration
        try:
            self._sample_and_project = torch.compile(
                self._sample_and_project_impl
            )
        except Exception:
            self._sample_and_project = self._sample_and_project_impl

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sample_plane(
        self, plane_features: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear sample from a 2-D plane.

        Args:
            plane_features: [1, rank, H, W]  (pre-permuted).
            coordinates:    [N, 2]  in [-1, 1].

        Returns:
            [N, rank]
        """
        grid = coordinates.view(1, 1, -1, 2)
        sampled = F.grid_sample(
            plane_features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.view(self.rank, -1).t()  # [N, rank]

    def _sample_and_project_impl(self, coords, plane_xy, plane_xz, plane_yz):
        """Core sampling + projection (compilable)."""
        # Permute planes to [1, rank, H, W] for grid_sample
        p_xy = plane_xy.permute(2, 0, 1).unsqueeze(0)
        p_xz = plane_xz.permute(2, 0, 1).unsqueeze(0)
        p_yz = plane_yz.permute(2, 0, 1).unsqueeze(0)

        xy_feat = self._sample_plane(p_xy, coords[:, :2])       # (x, y)
        xz_feat = self._sample_plane(p_xz, coords[:, [0, 2]])   # (x, z)
        yz_feat = self._sample_plane(p_yz, coords[:, 1:])        # (y, z)

        triplane = torch.cat([xy_feat, xz_feat, yz_feat], dim=1)  # [N, 3R]
        return self.projection(triplane)                           # [N, C]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: [N, 3] positions in [-1, 1]^3.

        Returns:
            [N, out_channels]  geometry features.
        """
        coords = coordinates.clamp(-1.0, 1.0)
        output = self._sample_and_project(
            coords, self.plane_xy, self.plane_xz, self.plane_yz
        )
        return output.clamp(-10.0, 10.0)
