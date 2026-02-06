"""
Encoder Factory for GlowGS

Always builds the full hybrid encoder (hash + VM + FiLM).  The ``enable_vm``
flag controls whether the VM branch contributes to the output:

    enable_vm=True  → full GlowGS  (hash + VM, FiLM modulated)
    enable_vm=False → VM output zeroed, numerically equivalent to hash-only
                      but the encoder object has the *same* interface / dims.

No separate ``hash_only`` or ``3dgs`` variant classes exist; ablation is done
via the single ``enable_vm`` boolean.
"""

import torch
import torch.nn as nn
import tinycudann as tcnn
from .geo_encoder import GeoEncoder
from .geometry_appearance_encoder import GeometryAppearanceEncoder


def create_gaussian_encoder(
    encoding_config: dict,
    network_config: dict,
    geo_resolution: int = 128,
    geo_rank: int = 48,
    geo_channels: int = 32,
    enable_vm: bool = True,
    *,
    # Legacy kwargs kept so old call-sites don't crash; values are ignored.
    variant: str | None = None,
    feature_mod_type: str | None = None,
):
    """
    Build the GlowGS hybrid encoder.

    Args:
        encoding_config: Hash grid config dict (for ``tcnn.Encoding``).
        network_config:  MLP config dict (unused here; kept for interface).
        geo_resolution:  VM initial resolution.
        geo_rank:        VM rank.
        geo_channels:    VM output channels.
        enable_vm:       If *False*, the VM branch output is zeroed inside the
                         ``GeometryAppearanceEncoder`` (bypass), so the encoder
                         degrades to hash-only numerically while keeping the
                         same output dimensions and interface.
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
        feature_role_split=True,
        use_film=True,           # FiLM is always structurally present
        enable_vm=enable_vm,     # ← bypass switch
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
        f"geo_dim={getattr(encoder, 'geo_dim', None)} "
        f"out_dim={encoder.n_output_dims}"
    )
    return encoder


def get_encoder_output_dims(encoder, variant: str | None = None):
    """
    Return (base_dim, geometry_dim, appearance_dim).

    For the hybrid encoder with role-split these are all equal to hash_dim.
    The ``variant`` arg is accepted but ignored (legacy compat).
    """
    base_dim = encoder.n_output_dims
    geo_dim = getattr(encoder, 'geometry_dim', base_dim)
    app_dim = getattr(encoder, 'appearance_dim', base_dim)
    return base_dim, geo_dim, app_dim
