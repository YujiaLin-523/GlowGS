"""
Feature Encoders for GlowGS

Provides geometry-appearance dual-branch encoding for 3D Gaussian Splatting:
- GeoEncoder: Geometry encoder using tri-plane factorization
- GeometryAppearanceEncoder: Dual-branch with explicit feature role split

Author: GlowGS Project
License: See LICENSE.md
"""

from .geo_encoder import GeoEncoder
from .geometry_appearance_encoder import GeometryAppearanceEncoder

__all__ = ['GeoEncoder', 'GeometryAppearanceEncoder']
