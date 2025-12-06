"""
Encoder Factory for Ablation Studies

Provides unified interface to create different encoder variants for systematic
ablation experiments on encoding architectures.
"""

import torch
import torch.nn as nn
import tinycudann as tcnn
from .geo_encoder import GeoEncoder
from .geometry_appearance_encoder import GeometryAppearanceEncoder


def create_gaussian_encoder(
    variant: str,
    encoding_config: dict,
    network_config: dict,
    geo_resolution: int = 48,
    geo_rank: int = 6,
    geo_channels: int = 8,
):
    """
    Factory function to create encoder based on variant type for ablation studies.
    
    Supports systematic comparison of:
    - Hybrid (hash + VM tri-plane): Paper's full approach
    - Hash-only: Discrete encoding baseline (LocoGS-style)
    - VM-only: Continuous encoding baseline (TensorRF-style)
    - No-role-split: Hybrid structure but single latent (ablate feature disentanglement)
    
    Args:
        variant: One of ["hybrid", "hash_only", "vm_only", "no_role_split"]
        encoding_config: Hash grid config (for hash_only and hybrid)
        network_config: MLP config (unused here, kept for interface consistency)
        geo_resolution: Tri-plane resolution for VM encoder
        geo_rank: Low-rank factorization rank
        geo_channels: VM encoder output channels
    
    Returns:
        encoder: Encoder instance with consistent interface
            - forward(coords) returns features or (shared, geometry, appearance) tuple
            - n_output_dims property for downstream head compatibility
    """
    
    if variant == "hybrid":
        # Paper default: Hash grid + VM tri-plane with geometry/appearance role split
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        encoder = GeometryAppearanceEncoder(
            hash_encoder=hash_encoder,
            out_dim=hash_encoder.n_output_dims,
            geo_channels=geo_channels,
            geo_resolution=geo_resolution,
            geo_rank=geo_rank,
            feature_role_split=True  # Enable disentanglement
        )
        return encoder
    
    elif variant == "hash_only":
        # Ablation: Only hash grid (discrete encoding), no VM component
        # Mimics LocoGS approach: multi-resolution hash without global priors
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        
        # Wrap in a simple adapter for interface consistency
        class HashOnlyEncoder(nn.Module):
            def __init__(self, hash_enc):
                super().__init__()
                self.hash_encoder = hash_enc
                self._n_output_dims = hash_enc.n_output_dims
            
            @property
            def n_output_dims(self):
                return self._n_output_dims
            
            def forward(self, coords):
                # Return single fused latent (no role split for hash-only)
                return self.hash_encoder(coords).float()
        
        return HashOnlyEncoder(hash_encoder)
    
    elif variant == "vm_only":
        # Ablation: Only VM/TensorRF tri-plane (continuous encoding), no hash
        # Tests whether smooth global priors alone suffice
        geo_encoder = GeoEncoder(
            resolution=geo_resolution,
            rank=geo_rank,
            out_channels=geo_channels,
            init_scale=0.01
        )
        
        # Wrap for interface consistency
        class VMOnlyEncoder(nn.Module):
            def __init__(self, geo_enc, output_dim):
                super().__init__()
                self.geo_encoder = geo_enc
                # Add projection to match expected output dimension
                self.projection = nn.Linear(geo_enc.out_channels, output_dim)
                nn.init.normal_(self.projection.weight, std=0.01)
                nn.init.zeros_(self.projection.bias)
                self._n_output_dims = output_dim
            
            @property
            def n_output_dims(self):
                return self._n_output_dims
            
            def forward(self, coords):
                geo_latent = self.geo_encoder(coords)
                return self.projection(geo_latent)
        
        # Match output dimension to hash grid for fair comparison
        hash_dim = encoding_config["n_levels"] * encoding_config["n_features_per_level"]
        return VMOnlyEncoder(geo_encoder, hash_dim)
    
    elif variant == "no_role_split":
        # Ablation: Hybrid structure (hash + VM) but no geometry/appearance disentanglement
        # Tests whether role split is necessary or just architectural capacity matters
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        encoder = GeometryAppearanceEncoder(
            hash_encoder=hash_encoder,
            out_dim=hash_encoder.n_output_dims,
            geo_channels=geo_channels,
            geo_resolution=geo_resolution,
            geo_rank=geo_rank,
            feature_role_split=False  # Disable disentanglement (fused latent)
        )
        return encoder
    
    else:
        raise ValueError(
            f"Unknown encoder_variant: {variant}. "
            f"Expected one of ['hybrid', 'hash_only', 'vm_only', 'no_role_split']"
        )


def get_encoder_output_dims(encoder, variant: str):
    """
    Get output dimensions for encoder variant (helper for downstream heads).
    
    Returns:
        (base_dim, geometry_dim, appearance_dim) tuple
        - For role-split encoders: geometry_dim and appearance_dim differ
        - For fused encoders: all three are equal
    """
    base_dim = encoder.n_output_dims
    
    if variant == "hybrid":
        # Role split: geometry and appearance have extra residual dimension
        if hasattr(encoder, 'geometry_dim'):
            return base_dim, encoder.geometry_dim, encoder.appearance_dim
        else:
            # Fallback if role split disabled
            return base_dim, base_dim, base_dim
    else:
        # All other variants use fused latent
        return base_dim, base_dim, base_dim
