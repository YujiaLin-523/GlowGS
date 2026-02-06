"""
Geometry-Appearance Dual-Branch Encoder (GlowGS Innovation #1)

Pipeline:
    xyz ──► Hash Grid ──► hash_latent  [N, H]  (high-freq appearance)
    xyz ──► VM GeoEncoder(AABB) ──► geo_feat [N, G]  (low-freq geometry)

    FiLM modulation (geo gates hash):
        geo_feat → MLP → (scale, shift)  each [N, H]
        geometry_latent  = hash * (1 + scale_g) + shift_g * hash_mag
        appearance_latent = hash * (1 + scale_a) + shift_a * hash_mag

    Output: (shared_latent, geometry_latent, appearance_latent)
            each [N, H]  → downstream MLP heads

Dimension contract (example):
    Hash dim H = 32 (tcnn: 16 levels × 2 features)
    VM   dim G = 32  (GeoEncoder out_channels)
    FiLM hidden = 128  (small MLP: G → 128 → 2H)
"""

import torch
import torch.nn as nn
from typing import Tuple, Union
from .geo_encoder import GeoEncoder


class GeometryAppearanceEncoder(nn.Module):
    """
    Hybrid encoder: Hash Grid (appearance) + VM GeoEncoder (geometry) + FiLM.

    Args:
        hash_encoder:       Pre-built tcnn.Encoding (n_input_dims=3).
        out_dim:            Output feature dim per branch (should == hash_dim).
        geo_channels:       GeoEncoder output channels (G).
        geo_resolution:     VM initial resolution.
        geo_rank:           VM rank.
        feature_role_split: Return 3-tuple or single fused tensor.
        use_film:           True = FiLM modulation, False = naive concat.
    """

    def __init__(
        self,
        hash_encoder,
        out_dim: int,
        geo_channels: int = 32,
        geo_resolution: int = 128,
        geo_rank: int = 48,
        feature_role_split: bool = True,
        use_film: bool = True,
    ):
        super().__init__()

        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims   # H
        self.use_film = use_film

        # VM geometry encoder (accepts raw xyz + aabb)
        self.geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.1,
        )
        self.geo_dim = geo_channels                  # G

        # ---- FiLM generators (geo → modulate hash) ----
        # Small MLP:  G → film_hidden → 2H  (scale + shift)
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

        # Init FiLM to identity: final layer weights/bias = 0
        # → scale=0, shift=0 → output = hash * 1 + 0 = hash
        for seq in [self.geometry_film, self.appearance_film]:
            nn.init.zeros_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)

        # ---- Fallback concat layers ----
        self.fusion_layer = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)
        self.geometry_proj_concat = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)
        self.appearance_proj_concat = nn.Linear(self.hash_dim + self.geo_dim, out_dim, bias=True)
        nn.init.normal_(self.fusion_layer.weight, std=0.01)
        nn.init.zeros_(self.fusion_layer.bias)

        self._out_dim = out_dim
        self.feature_role_split = feature_role_split

        # ---- Compilation ----
        try:
            self._forward_film_compiled = torch.compile(self._forward_film_impl)
        except Exception:
            self._forward_film_compiled = self._forward_film_impl

    # ---------- warmup ----------
    def set_warmup_progress(self, progress: float):
        """FiLM warmup (0→1). Called each iteration from train.py."""
        self._warmup_progress = max(0.0, min(1.0, progress))

    # ---------- dimension properties ----------
    @property
    def n_output_dims(self) -> int:
        return self._out_dim

    @property
    def role_dim(self) -> int:
        return 0

    @property
    def geometry_dim(self) -> int:
        return self._out_dim

    @property
    def appearance_dim(self) -> int:
        return self._out_dim

    # ---------- FiLM forward ----------
    def _forward_film_impl(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        """FiLM modulation: geo generates scale/shift for hash features."""
        g_params = self.geometry_film(geo_latent)        # [N, 2H]
        a_params = self.appearance_film(geo_latent)      # [N, 2H]

        g_scale, g_shift = g_params.chunk(2, dim=-1)     # each [N, H]
        a_scale, a_shift = a_params.chunk(2, dim=-1)

        # Characteristic magnitude of hash (detached to prevent feedback)
        hash_mag = hash_latent.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-3)

        warmup = getattr(self, '_warmup_progress', 1.0)

        # FiLM:  feat = hash * (1 + w·scale) + w·shift·hash_mag
        geometry_latent   = hash_latent * (1.0 + warmup * g_scale) + (warmup * g_shift * hash_mag)
        appearance_latent = hash_latent * (1.0 + warmup * a_scale) + (warmup * a_shift * hash_mag)

        shared_latent = hash_latent

        # Stability clamp
        geometry_latent   = torch.clamp(geometry_latent,   -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)
        shared_latent     = torch.clamp(shared_latent,     -10.0, 10.0)

        return shared_latent, geometry_latent, appearance_latent

    # ---------- concat forward (ablation baseline) ----------
    def _forward_concat(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        combined = torch.cat([hash_latent, geo_latent], dim=-1)
        geometry_latent   = self.geometry_proj_concat(combined)
        appearance_latent = self.appearance_proj_concat(combined)
        shared_latent     = self.fusion_layer(combined)

        geometry_latent   = torch.clamp(geometry_latent,   -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)
        shared_latent     = torch.clamp(shared_latent,     -10.0, 10.0)

        return shared_latent, geometry_latent, appearance_latent

    # ---------- dispatch ----------
    def _forward_split(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        if self.use_film:
            return self._forward_film_compiled(hash_latent, geo_latent)
        else:
            return self._forward_concat(hash_latent, geo_latent)

    # ---------- progressive upsample (delegate) ----------
    def upsample_resolution(self, new_resolution: int):
        """Upsample VM parameters of the inner GeoEncoder."""
        self.geo_encoder.upsample_resolution(new_resolution)

    # ---------- public forward ----------
    def forward(
        self,
        coordinates: torch.Tensor,
        aabb: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            coordinates: [N, 3] raw world positions.
            aabb:        [6]    scene bounding box.
        Returns:
            feature_role_split=True  → (shared, geometry, appearance)  each [N, H]
            feature_role_split=False → fused [N, out_dim]
        """
        # Hash branch (high-freq appearance).  tcnn outputs fp16 → cast to fp32.
        hash_latent = self.hash_encoder(coordinates).float()

        # VM branch (low-freq geometry, with built-in L∞ contraction)
        geo_latent = self.geo_encoder(coordinates, aabb)

        if self.feature_role_split:
            return self._forward_split(hash_latent, geo_latent)
        else:
            combined = torch.cat([hash_latent, geo_latent], dim=1)
            fused = self.fusion_layer(combined)
            return torch.clamp(fused, -10.0, 10.0)