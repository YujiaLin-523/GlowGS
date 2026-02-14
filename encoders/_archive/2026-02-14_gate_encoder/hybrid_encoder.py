import torch
import torch.nn as nn
from typing import Tuple

from .geo_encoder import GeoEncoder


class GeometryAppearanceEncoder(nn.Module):
    """
    Hybrid encoder: Hash Grid + tri-plane GeoEncoder + FiLM modulation.

    Provides explicit geometry / appearance disentanglement via
    Feature-wise Linear Modulation (FiLM).  Low-frequency geometry
    features modulate high-frequency hash features into two
    specialised branches — one for geometric attributes (opacity,
    scale, rotation) and one for appearance (SH / colour).

    Args:
        hash_encoder:    Pre-built ``tcnn.Encoding`` (n_input_dims=3).
        out_dim:         Per-branch output feature dimension (== hash_dim).
        geo_channels:    GeoEncoder output channels.
        geo_resolution:  Tri-plane spatial resolution.
        geo_rank:        Low-rank factorisation rank.
        enable_vm:       If *False*, GeoEncoder + FiLM are bypassed
                         (clean hash-only ablation).
    """

    def __init__(
        self,
        hash_encoder,
        out_dim: int,
        geo_channels: int = 8,
        geo_resolution: int = 96,
        geo_rank: int = 12,
        enable_vm: bool = True,
    ):
        super().__init__()

        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims  # H
        self.enable_vm = enable_vm

        # ── GeoEncoder (tri-plane, low-freq geometry) ────────────────────
        self.geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.01,
        )
        self.geo_dim = geo_channels  # G

        # ── FiLM generators: G → hidden → 2H  (scale + shift) ───────────
        film_hidden = max(128, self.hash_dim * 2)

        self.geometry_film = nn.Sequential(
            nn.Linear(self.geo_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, self.hash_dim * 2),
        )
        self.appearance_film = nn.Sequential(
            nn.Linear(self.geo_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, self.hash_dim * 2),
        )

        # Identity init: scale=0, shift=0  →  output = hash at start
        for seq in (self.geometry_film, self.appearance_film):
            nn.init.zeros_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)

        self._out_dim = out_dim

        # Optional torch.compile acceleration
        try:
            self._forward_film_compiled = torch.compile(
                self._forward_film_impl
            )
        except Exception:
            self._forward_film_compiled = self._forward_film_impl

    # ── dimension properties (match tcnn interface) ──────────────────────
    @property
    def n_output_dims(self) -> int:
        return self._out_dim

    @property
    def geometry_dim(self) -> int:
        return self._out_dim

    @property
    def appearance_dim(self) -> int:
        return self._out_dim

    # ── FiLM core ────────────────────────────────────────────────────────
    def _forward_film_impl(
        self,
        hash_latent: torch.Tensor,
        geo_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FiLM modulation: geo generates (scale, shift) for hash features."""
        g_params = self.geometry_film(geo_latent)    # [N, 2H]
        a_params = self.appearance_film(geo_latent)  # [N, 2H]

        g_scale, g_shift = g_params.chunk(2, dim=-1)  # each [N, H]
        a_scale, a_shift = a_params.chunk(2, dim=-1)

        # FiLM:  feat = hash * (1 + scale) + shift
        geometry_latent = hash_latent * (1.0 + g_scale) + g_shift
        appearance_latent = hash_latent * (1.0 + a_scale) + a_shift
        shared_latent = hash_latent

        # Stability clamp
        geometry_latent = torch.clamp(geometry_latent, -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)
        shared_latent = torch.clamp(shared_latent, -10.0, 10.0)

        return shared_latent, geometry_latent, appearance_latent

    # ── public forward ───────────────────────────────────────────────────
    def forward(
        self, coordinates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coordinates: [N, 3] contracted positions in [-1, 1]^3.

        Returns:
            (shared_latent, geometry_latent, appearance_latent)

            enable_vm=True:
                shared = clamped hash,
                geometry / appearance = FiLM-modulated hash
            enable_vm=False:
                all three = clamped hash  (GeoEncoder + FiLM bypassed)
        """
        # Hash branch (high-freq).  tcnn outputs fp16 → cast to fp32.
        hash_latent = self.hash_encoder(coordinates).float()

        if not self.enable_vm:
            h = torch.clamp(hash_latent, -10.0, 10.0)
            return h, h, h

        # Geo branch (low-freq)
        geo_latent = self.geo_encoder(coordinates)

        return self._forward_film_compiled(hash_latent, geo_latent)
