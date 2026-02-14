"""
Encoder Factory for GlowGS

Builds the hybrid encoder (Hash Grid + tri-plane GeoEncoder + FiLM).
The ``enable_vm`` flag controls whether the geometry branch contributes:

    enable_vm=True   → full GlowGS  (Hash + GeoEncoder + FiLM modulation)
    enable_vm=False  → GeoEncoder + FiLM bypassed; output = raw hash features
                       (clean hash-only ablation, equivalent to LocoGS baseline)

Author: GlowGS Project
License: See LICENSE.md
"""

import tinycudann as tcnn

from .hybrid_encoder import GeometryAppearanceEncoder


def create_gaussian_encoder(
    encoding_config: dict,
    network_config: dict,           # kept for interface consistency
    geo_resolution: int = 96,
    geo_rank: int = 12,
    geo_channels: int = 8,
    enable_vm: bool = True,
):
    """
    Build the GlowGS hybrid encoder.

    Args:
        encoding_config:  Hash grid config dict (for ``tcnn.Encoding``).
        network_config:   MLP config dict (unused; kept for interface).
        geo_resolution:   Tri-plane spatial resolution.
        geo_rank:         Low-rank factorisation rank per plane.
        geo_channels:     GeoEncoder output channels (G).
        enable_vm:        If *False*, GeoEncoder + FiLM are bypassed.

    Returns:
        ``GeometryAppearanceEncoder`` instance.
    """
    hash_encoder = tcnn.Encoding(
        n_input_dims=3, encoding_config=encoding_config
    )

    encoder = GeometryAppearanceEncoder(
        hash_encoder=hash_encoder,
        out_dim=hash_encoder.n_output_dims,
        geo_channels=geo_channels,
        geo_resolution=geo_resolution,
        geo_rank=geo_rank,
        enable_vm=enable_vm,
    )

    vm_tag = "ON" if enable_vm else "OFF(bypass)"
    print(
        f"[Encoder] enable_vm={enable_vm}({vm_tag}) "
        f"hash_dim={hash_encoder.n_output_dims} "
        f"geo_dim={encoder.geo_dim} "
        f"out_dim={encoder.n_output_dims} "
        f"(geometry_dim={encoder.geometry_dim} "
        f"appearance_dim={encoder.appearance_dim})"
    )
    return encoder


def get_encoder_output_dims(encoder):
    """
    Return (base_dim, geometry_dim, appearance_dim).

    With FiLM modulation all three equal hash_dim (= 32).
    """
    base_dim = encoder.n_output_dims
    geo_dim = getattr(encoder, "geometry_dim", base_dim)
    app_dim = getattr(encoder, "appearance_dim", base_dim)
    return base_dim, geo_dim, app_dim
