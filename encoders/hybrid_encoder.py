"""
Geometry-Appearance Dual-Branch Encoder (GlowGS Innovation #1)

Pipeline (enable_vm=True):
    xyz ──► Hash Grid ──► hash_latent  [N, H]  (high-freq features)
    xyz ──► VM GeoEncoder(AABB) ──► geo_vm [N, G]  (low-freq geometry)

    Concat for geometry (direct feature access):
        geometry_latent = concat(hash_latent, geo_vm)  [N, H+G]

    FiLM for appearance (geo guides appearance):
        geo_vm → MLP → (scale, shift)  each [N, H]
        appearance_latent = hash * (1 + w·scale) + w·shift·hash_mag  [N, H]

    Output: (geometry_latent, appearance_latent)
            geometry [N, H+G] → opacity/scaling/rotation heads
            appearance [N, H] → features_rest head

Pipeline (enable_vm=False, hash-only ablation):
    xyz ──► Hash Grid ──► hash_latent  [N, H]
    Output: (hash_latent, hash_latent)
    VM and FiLM are structurally present but completely bypassed.

Dimension contract (example):
    Hash dim H = 32 (tcnn: 16 levels × 2 features)
    VM   dim G = 32  (GeoEncoder out_channels)
    Geometry output = H+G = 64
    Appearance output = H = 32
    FiLM hidden = 128  (small MLP: G → 128 → 2H)
"""

import torch
import torch.nn as nn
from typing import Tuple
from .vm_encoder import GeoEncoder


class GeometryAppearanceEncoder(nn.Module):
    """
    Hybrid encoder: Hash Grid (appearance) + VM GeoEncoder (geometry) + FiLM.

    Args:
        hash_encoder:       Pre-built tcnn.Encoding (n_input_dims=3).
        out_dim:            Output feature dim per branch (should == hash_dim).
        geo_channels:       GeoEncoder output channels (G).
        geo_resolution:     VM initial resolution.
        geo_rank:           VM rank.
        enable_vm:          If False, VM branch and FiLM are bypassed
                            (clean hash-only ablation).
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
        self.hash_dim = hash_encoder.n_output_dims   # H
        self.enable_vm = enable_vm                    # ← ablation bypass

        # VM geometry encoder (accepts raw xyz + aabb)
        self.geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.1,
        )
        self.geo_dim = geo_channels                  # G

        # ---- FiLM generator for appearance only (geo → modulate hash) ----
        # Geometry uses direct concat instead of FiLM
        # Small MLP:  G → film_hidden → 2H  (scale + shift)
        film_hidden = max(128, self.hash_dim * 2)
        self.appearance_film = nn.Sequential(
            nn.Linear(self.geo_dim, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, self.hash_dim * 2),
        )

        # Init FiLM to identity: final layer weights/bias = 0
        # → scale=0, shift=0 → output = hash * 1 + 0 = hash
        nn.init.zeros_(self.appearance_film[-1].weight)
        nn.init.zeros_(self.appearance_film[-1].bias)

        # Output dimensions: geometry = H+G, appearance = H
        self._geometry_dim = self.hash_dim + self.geo_dim  # H+G
        self._appearance_dim = self.hash_dim                 # H
        self._out_dim = out_dim

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
        """Return hash_dim for backward compatibility (used by factory checks)."""
        return self.hash_dim

    @property
    def geometry_dim(self) -> int:
        """Geometry branch output dimension: H+G when VM enabled, H otherwise."""
        return self._geometry_dim if self.enable_vm else self.hash_dim

    @property
    def appearance_dim(self) -> int:
        """Appearance branch output dimension: always H."""
        return self.hash_dim

    # ---------- Forward implementations ----------
    def _forward_film_impl(self, hash_latent: torch.Tensor, geo_latent: torch.Tensor):
        """Concat for geometry, FiLM for appearance."""
        # Geometry: direct concat (no information bottleneck)
        geometry_latent = torch.cat([hash_latent, geo_latent], dim=-1)  # [N, H+G]

        # Appearance: FiLM modulation (geo guides appearance)
        a_params = self.appearance_film(geo_latent)  # [N, 2H]
        a_scale, a_shift = a_params.chunk(2, dim=-1)

        # Characteristic magnitude of hash (detached to prevent feedback)
        hash_mag = hash_latent.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-3)

        warmup = getattr(self, '_warmup_progress', 1.0)

        # FiLM:  appearance = hash * (1 + w·scale) + w·shift·hash_mag
        appearance_latent = hash_latent * (1.0 + warmup * a_scale) + (warmup * a_shift * hash_mag)

        # Stability clamp
        geometry_latent = torch.clamp(geometry_latent, -10.0, 10.0)
        appearance_latent = torch.clamp(appearance_latent, -10.0, 10.0)

        return geometry_latent, appearance_latent

    # ---------- progressive upsample (delegate) ----------
    def upsample_resolution(self, new_resolution: int):
        """Upsample VM parameters of the inner GeoEncoder."""
        self.geo_encoder.upsample_resolution(new_resolution)

    # ---------- public forward ----------
    def forward(
        self,
        coordinates: torch.Tensor,
        aabb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coordinates: [N, 3] raw world positions.
            aabb:        [6]    scene bounding box.
        Returns:
            (geometry_latent, appearance_latent)

            enable_vm=True:  geometry = concat(hash, geo_vm) [N, H+G]
                             appearance = FiLM(hash, geo_vm) [N, H]
            enable_vm=False: both = clamped raw hash [N, H] (VM & FiLM bypassed)
        """
        # Hash branch (high-freq features).  tcnn outputs fp16 → cast to fp32.
        hash_latent = self.hash_encoder(coordinates).float()

        # ---- enable_vm=False fast path: skip VM + FiLM entirely ----
        # This is a clean hash-only ablation: no VM computation, no FiLM,
        # both outputs are identical raw hash features.
        if not self.enable_vm:
            h = torch.clamp(hash_latent, -10.0, 10.0)
            return h, h

        # ---- enable_vm=True: full pipeline ----
        geo_latent = self.geo_encoder(coordinates, aabb)
        return self._forward_film_compiled(hash_latent, geo_latent)