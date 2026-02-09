"""
Feature Encoders for GlowGS

Provides VM scaffold + Hash residual encoding for 3D Gaussian Splatting:
- vm_encoder.GeoEncoder:  VM (Vector-Matrix) tri-plane geometry encoder
- hybrid_encoder.GeometryAppearanceEncoder:  VM scaffold + Hash residual + alignment
- factory:  Unified creation with ``enable_vm`` ablation switch

Author: GlowGS Project
License: See LICENSE.md
"""

from .vm_encoder import GeoEncoder
from .hybrid_encoder import GeometryAppearanceEncoder
from .factory import create_gaussian_encoder, get_encoder_output_dims

__all__ = [
    'GeoEncoder',
    'GeometryAppearanceEncoder',
    'create_gaussian_encoder',
    'get_encoder_output_dims',
]
