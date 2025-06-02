"""
Hexahedral Element Quality Metrics

A Python package for computing quality metrics of 8-node hexahedral finite elements.
This package provides various geometric quality measures commonly used in mesh generation
and finite element analysis.

Quality Metrics Available:
- aspect_ratio: Ratio of maximum to minimum edge length
- warpage: Maximum face planarity deviation (degrees)
- skewness: Maximum deviation from 90° face angles (degrees)
- orthogonal_quality: Measure of edge orthogonality at vertices [0,1]
- scaled_jacobian: Normalized jacobian quality metric [0,1]
- jacobian: Raw jacobian determinant of affine mapping

All functions expect hexahedral vertices in standard FEM ordering:
- Vertices 0-3: bottom face (counter-clockwise from outside)
- Vertices 4-7: top face (counter-clockwise from outside)
- Vertical edges: 0-4, 1-5, 2-6, 3-7

Example:
    >>> import hex_quality as hq
    >>> vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0],
    ...             [0,0,1], [1,0,1], [1,1,1], [0,1,1]]
    >>> hq.aspect_ratio(vertices)
    1.0
    >>> hq.orthogonal_quality(vertices)
    1.0
"""

from .metrics import (
    aspect_ratio,
    warpage,
    skewness,
    orthogonal_quality,
    scaled_jacobian,
    jacobian
)

__version__ = "1.0.0"
__author__ = "Finite Element Quality Assessment"

__all__ = [
    'aspect_ratio',
    'warpage',
    'skewness',
    'orthogonal_quality',
    'scaled_jacobian',
    'jacobian'
]