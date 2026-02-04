"""
Hash-only encoder for ablation: retains hash grid pathway without VM/tri-plane branch.
Provides the same output interface as the hybrid encoder so downstream heads remain unchanged.
"""

import torch
import torch.nn as nn


class HashOnlyEncoder(nn.Module):
    def __init__(self, hash_encoder, feature_role_split: bool = True):
        super().__init__()
        self.hash_encoder = hash_encoder
        self.hash_dim = hash_encoder.n_output_dims
        self._out_dim = self.hash_dim
        self.feature_role_split = feature_role_split
        # TODO(stage1-task3): ensure hash_only output interface matches hybrid to keep downstream identical

    @property
    def n_output_dims(self) -> int:
        return self._out_dim

    @property
    def geometry_dim(self) -> int:
        return self._out_dim

    @property
    def appearance_dim(self) -> int:
        return self._out_dim

    def set_warmup_progress(self, progress: float):
        # Hash-only path has no VM to warm up; track for interface parity.
        self._warmup_progress = max(0.0, min(1.0, progress))

    def forward(self, coordinates: torch.Tensor):
        # Hash grid branch only
        hash_latent = self.hash_encoder(coordinates).float()
        hash_latent = self._stabilize(hash_latent)
        if self.feature_role_split:
            # Shared/geometry/appearance all reuse the same hash latent
            return hash_latent, hash_latent, hash_latent
        return hash_latent

    def _stabilize(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp activations for numerical stability (mirror hybrid behaviour)
        return torch.clamp(x, min=-10.0, max=10.0)
