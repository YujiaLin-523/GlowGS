"""
Encoder Factory for GlowGS

Always builds the full hybrid encoder (hash + VM + FiLM).  The ``enable_vm``
flag controls whether the VM branch contributes to the output:

    enable_vm=True  → full GlowGS  (hash + VM, FiLM modulated)
    enable_vm=False → VM & FiLM bypassed; all three outputs = raw hash
                      features (clean hash-only ablation)

Ablation is controlled by the single ``enable_vm`` boolean.
"""

import torch
import torch.nn as nn
import tinycudann as tcnn
from .hybrid_encoder import GeometryAppearanceEncoder


def create_gaussian_encoder(
    encoding_config: dict,
    network_config: dict,
    geo_resolution: int = 128,
    geo_rank: int = 48,
    geo_channels: int = 32,
    enable_vm: bool = True,
):
    """
    Build the GlowGS hybrid encoder.

    Args:
        encoding_config: Hash grid config dict (for ``tcnn.Encoding``).
        network_config:  MLP config dict (unused here; kept for interface).
        geo_resolution:  VM initial resolution.
        geo_rank:        VM rank.
        geo_channels:    VM output channels.
        enable_vm:       If *False*, the VM branch and FiLM are bypassed,
                         and the encoder outputs raw hash features for all
                         three branches (clean hash-only ablation).
    Returns:
        encoder – ``GeometryAppearanceEncoder`` instance.
    """
    hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)

    encoder = GeometryAppearanceEncoder(
        hash_encoder=hash_encoder,
        out_dim=hash_encoder.n_output_dims,
        geo_channels=geo_channels,
        geo_resolution=geo_resolution,
        geo_rank=geo_rank,
        enable_vm=enable_vm,
    )

    if encoder.n_output_dims != hash_encoder.n_output_dims:
        raise ValueError(
            f"Encoder dim mismatch: out_dim={encoder.n_output_dims} "
            f"vs hash_dim={hash_encoder.n_output_dims}"
        )
    vm_tag = "ON" if enable_vm else "OFF(bypass)"
    print(
        f"[Encoder] enable_vm={enable_vm}({vm_tag}) "
        f"hash_dim={hash_encoder.n_output_dims} "
        f"geo_vm_dim={getattr(encoder, 'geo_dim', None)} "
        f"geometry_dim={encoder.geometry_dim} "
        f"appearance_dim={encoder.appearance_dim}"
    )
    return encoder


def get_encoder_output_dims(encoder):
    """
    Return (base_dim, geometry_dim, appearance_dim).

    For the hybrid encoder with concat architecture:
        - base_dim = hash_dim (H)
        - geometry_dim = H+G when enable_vm=True, H otherwise
        - appearance_dim = H (always)
    """
    base_dim = encoder.n_output_dims
    geo_dim = getattr(encoder, 'geometry_dim', base_dim)
    app_dim = getattr(encoder, 'appearance_dim', base_dim)
    return base_dim, geo_dim, app_dim
