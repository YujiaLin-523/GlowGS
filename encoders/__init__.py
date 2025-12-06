"""
Feature Encoders for GlowGS

Provides geometry-appearance dual-branch encoding for 3D Gaussian Splatting:
- GeoEncoder: Geometry encoder using tri-plane factorization
- GeometryAppearanceEncoder: Dual-branch with explicit feature role split
- encoder_factory: Unified interface for ablation studies

Author: GlowGS Project
License: See LICENSE.md
"""

from .geo_encoder import GeoEncoder
from .geometry_appearance_encoder import GeometryAppearanceEncoder
from .encoder_factory import create_gaussian_encoder, get_encoder_output_dims

__all__ = [
    'GeoEncoder', 
    'GeometryAppearanceEncoder',
    'create_gaussian_encoder',
    'get_encoder_output_dims'
]
