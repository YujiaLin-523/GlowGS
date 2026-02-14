"""
Feature Encoders for GlowGS

Provides geometry-appearance dual-branch encoding for 3D Gaussian Splatting:
- geo_encoder.GeoEncoder:  Tri-plane factorisation geometry encoder
- hybrid_encoder.GeometryAppearanceEncoder:  Hash + GeoEncoder + FiLM modulation
- factory:  Unified creation with ``enable_vm`` ablation switch

Author: GlowGS Project
License: See LICENSE.md
"""

from .geo_encoder import GeoEncoder
from .hybrid_encoder import GeometryAppearanceEncoder
from .factory import create_gaussian_encoder, get_encoder_output_dims

__all__ = [
    "GeoEncoder",
    "GeometryAppearanceEncoder",
    "create_gaussian_encoder",
    "get_encoder_output_dims",
]
