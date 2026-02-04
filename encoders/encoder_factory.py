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
from .hash_only_encoder import HashOnlyEncoder


def create_gaussian_encoder(
    variant: str,
    encoding_config: dict,
    network_config: dict,
    geo_resolution: int = 48,
    geo_rank: int = 6,
    geo_channels: int = 8,
    feature_mod_type: str = "film",
):
    """
    Factory function to create encoder based on variant type.
    
    Supports two encoder types for ablation studies:
    - 'hybrid': Full GlowGS hybrid encoder (hash + VM tri-plane with geometry/appearance split)
    - '3dgs': No encoder (uses explicit SH parameters, 3DGS baseline mode)
    
    Args:
        variant: Either "hybrid" or "3dgs"
        encoding_config: Hash grid config (for hybrid)
        network_config: MLP config (unused here, kept for interface consistency)
        geo_resolution: Tri-plane resolution for VM encoder
        geo_rank: Low-rank factorization rank
        geo_channels: VM encoder output channels
        feature_mod_type: 'film' (FiLM modulation) or 'concat' (naive concatenation)
    
    Returns:
        encoder: Encoder instance with consistent interface
            - forward(coords) returns features or (shared, geometry, appearance) tuple
            - n_output_dims property for downstream head compatibility
    """
    
    if variant == "hybrid":
        # Full GlowGS: Hash grid + VM tri-plane with geometry/appearance role split
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        # Determine if FiLM modulation is enabled (feature_mod_type='film')
        use_film = (feature_mod_type == 'film')
        encoder = GeometryAppearanceEncoder(
            hash_encoder=hash_encoder,
            out_dim=hash_encoder.n_output_dims,
            geo_channels=geo_channels,
            geo_resolution=geo_resolution,
            geo_rank=geo_rank,
            feature_role_split=True,  # Enable disentanglement
            use_film=use_film,        # FiLM vs concat mode
        )
        # TODO(stage1-task3): assert hash_only out_dim == hybrid out_dim under current fusion/role-split settings
        if encoder.n_output_dims != hash_encoder.n_output_dims:
            raise ValueError(
                f"Hybrid encoder dim mismatch: out_dim={encoder.n_output_dims} vs hash_dim={hash_encoder.n_output_dims}"
            )
        print(f"[Encoder] variant=hybrid mode={'film' if use_film else 'concat'} hash_dim={hash_encoder.n_output_dims} geo_dim={getattr(encoder, 'geo_dim', None)} out_dim={encoder.n_output_dims}")
        return encoder

    elif variant == "hash_only":
        # Hash-only ablation: reuse hash grid without VM planes or gates
        # TODO(stage1-task3): encoder_factory must be single source of truth for encoder_variant
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        encoder = HashOnlyEncoder(
            hash_encoder=hash_encoder,
            feature_role_split=True,
        )
        if encoder.n_output_dims != hash_encoder.n_output_dims:
            raise ValueError(
                f"Hash-only encoder dim mismatch: out_dim={encoder.n_output_dims} vs hash_dim={hash_encoder.n_output_dims}"
            )
        print(f"[Encoder] variant=hash_only hash_dim={hash_encoder.n_output_dims} out_dim={encoder.n_output_dims}")
        return encoder
    
    elif variant == "3dgs":
        # 3DGS baseline mode: No encoder at all
        # Uses explicit per-Gaussian SH parameters instead of coordinate-based encoding
        # Returns dummy features to maintain interface compatibility with rendering pipeline
        
        class DummyEncoder(nn.Module):
            """
            Dummy encoder for 3DGS baseline mode.
            Returns zero features, signaling that explicit SH parameters should be used.
            This maintains API compatibility while bypassing the encoder-based representation.
            """
            def __init__(self):
                super().__init__()
                # Minimal output dimension (unused, but needed for interface)
                self._n_output_dims = 1
            
            @property
            def n_output_dims(self):
                return self._n_output_dims
            
            def forward(self, coords):
                # Return zero features (signals to use explicit SH params)
                batch_size = coords.shape[0]
                return torch.zeros((batch_size, self._n_output_dims), 
                                 dtype=coords.dtype, device=coords.device)
        
        return DummyEncoder()
    
    else:
        raise ValueError(
            f"Unknown encoder_variant: {variant}. "
            f"Expected 'hybrid', 'hash_only' or '3dgs'"
        )


def get_encoder_output_dims(encoder, variant: str):
    """
    Get output dimensions for encoder variant (helper for downstream heads).
    
    Returns:
        (base_dim, geometry_dim, appearance_dim) tuple
        - For role-split encoders: geometry_dim and appearance_dim differ
        - For fused encoders: all three are equal
        - For 3dgs variant: returns minimal dims (encoder is dummy)
    """
    base_dim = encoder.n_output_dims
    
    if variant == "hybrid":
        # Role split: geometry and appearance have extra residual dimension
        if hasattr(encoder, 'geometry_dim'):
            return base_dim, encoder.geometry_dim, encoder.appearance_dim
        else:
            # Fallback if role split disabled
            return base_dim, base_dim, base_dim
    elif variant == "hash_only":
        return base_dim, base_dim, base_dim
    elif variant == "3dgs":
        # 3DGS mode: encoder is dummy, dimensions don't matter
        # (explicit SH parameters are used instead)
        return base_dim, base_dim, base_dim
    else:
        # All other variants use fused latent
        return base_dim, base_dim, base_dim
