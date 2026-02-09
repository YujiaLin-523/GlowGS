"""
VM Scaffold + Hash Residual Encoder (with Lightweight Alignment)

Architecture
    coordinates  [N, 3]  in [0, 1]^3  (from contract_to_unisphere)
    ├── Hash branch:  fh = Hash(coordinates)  ∈ R^H  (high-freq detail)
    └── VM branch:    fg = VM(coordinates)    ∈ R^G  (low-freq scaffold)

    Residual alignment (applied to Hash residual only):
        r̃ = W · fh          (W: linear projection, identity-init)
        r  = c ⊙ r̃          (c: learnable channel scale, ones-init)

    Fusion (no concat, no gate, no FiLM):
        f = fg + r

    Output: single fused feature  f ∈ R^32  → all four MLP heads
            (opacity, scaling, rotation, SH/color)

    Ablation:
        enable_vm=True   →  full VM scaffold + Hash residual (default)
        enable_vm=False  →  hash-only:  f = Hash(coordinates)

Dimension contract:
    Hash dim H = 32  (tcnn: 16 levels × 2 features)
    VM   dim G = 32  (GeoEncoder out_channels)
    Fused output = 32  (unified for all heads)

Both branches receive the **exact same** ``coordinates`` tensor.
Hash (tcnn) uses [0,1]^3 directly.
VM (GeoEncoder) internally converts to [-1,1]^3 for grid_sample — this
is a PyTorch API requirement, not a different normalization.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .vm_encoder import GeoEncoder


class GeometryAppearanceEncoder(nn.Module):
    """
    Hybrid encoder: VM scaffold + Hash residual with lightweight alignment.

    Args:
        hash_encoder:       Pre-built tcnn.Encoding (n_input_dims=3).
        out_dim:            Output feature dim (should == hash_dim == geo_channels).
        geo_channels:       GeoEncoder output channels (G), must == hash_dim.
        geo_resolution:     VM initial resolution.
        geo_rank:           VM rank.
        enable_vm:          If False, VM branch is bypassed (hash-only ablation).
    """

    def __init__(
        self,
        hash_encoder,
        out_dim: int,
        geo_channels: int = 32,
        geo_resolution: int = 128,
        geo_rank: int = 48,
        enable_vm: bool = True,
    ):
        super().__init__()

        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims   # H (typically 32)
        self.enable_vm = enable_vm

        # Dimension consistency: VM and Hash must produce same dim for residual
        assert geo_channels == self.hash_dim, (
            f"VM geo_channels ({geo_channels}) must equal hash_dim ({self.hash_dim}) "
            f"for residual fusion."
        )

        # ── VM geometry encoder (low-freq scaffold) ──────────────────────
        self.geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.1,
        )
        self.geo_dim = geo_channels   # G == H

        # ── Residual alignment module (Hash → aligned residual) ──────────
        # Linear projection: identity-init so initial r̃ ≈ fh
        self.linear_align = nn.Linear(self.hash_dim, self.hash_dim, bias=False)
        nn.init.eye_(self.linear_align.weight)

        # Channel scale: ones-init so initial r = r̃ (no amplitude change)
        self.channel_scale = nn.Parameter(torch.ones(self.hash_dim))

        # ── Output dimension (unified) ───────────────────────────────────
        self._out_dim = out_dim
        self._fused_dim = self.hash_dim   # == geo_channels == 32

    # ---------- dimension properties ----------
    @property
    def n_output_dims(self) -> int:
        return self._fused_dim

    @property
    def geometry_dim(self) -> int:
        return self._fused_dim

    @property
    def appearance_dim(self) -> int:
        return self._fused_dim

    # ---------- progressive upsample (delegate to GeoEncoder) ----------
    def upsample_resolution(self, new_resolution: int):
        self.geo_encoder.upsample_resolution(new_resolution)

    # ---------- public forward ----------
    def forward(self, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coordinates: [N, 3] positions in [0, 1]^3.
                         Produced by ``contract_to_unisphere()`` upstream.
                         Passed to Hash and VM identically (same tensor).

        Returns:
            (fused, fused) — same tensor twice for API compatibility.
        """
        # ── Hash branch ──────────────────────────────────────────────────
        # tcnn outputs fp16 → cast to fp32 for stability
        hash_latent = self.hash_encoder(coordinates).float()   # [N, H]

        if not self.enable_vm:
            return hash_latent, hash_latent

        # ── VM branch (same coordinates tensor) ─────────────────────────
        fg = self.geo_encoder(coordinates)                     # [N, G=32]

        # ── Residual alignment ───────────────────────────────────────────
        r = self.channel_scale * self.linear_align(hash_latent)  # [N, H]

        # ── Fusion ───────────────────────────────────────────────────────
        f = fg + r                                             # [N, 32]

        return f, f
