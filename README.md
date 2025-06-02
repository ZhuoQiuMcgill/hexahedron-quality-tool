# Hex Quality

A Python package for computing quality metrics of 8-node hexahedral finite elements. This package provides various geometric quality measures commonly used in mesh generation and finite element analysis.

## Features

- **Aspect Ratio**: Ratio of maximum to minimum edge length
- **Warpage**: Maximum face planarity deviation (degrees)  
- **Skewness**: Maximum deviation from 90° face angles (degrees)
- **Orthogonal Quality**: Measure of edge orthogonality at vertices [0,1]
- **Scaled Jacobian**: Normalized jacobian quality metric [0,1] 
- **Jacobian**: Raw jacobian determinant of affine mapping

## Installation

### From source
```bash
git clone https://github.com/ZhuoQiuMcgill/hexahedron-quality-tool.git
cd hex_quality
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev,test]"
```

## Requirements

- Python 3.7+
- NumPy >= 1.19.0

## Quick Start

```python
import hex_quality as hq

# Define a unit cube (perfect hexahedron)
vertices = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
]

# Compute quality metrics
aspect_ratio = hq.aspect_ratio(vertices)        # 1.0 (perfect)
warpage = hq.warpage(vertices)                  # 0.0° (planar faces)
skewness = hq.skewness(vertices)                # 0.0° (right angles)
ortho_quality = hq.orthogonal_quality(vertices) # 1.0 (perfect orthogonality)

print(f"Aspect Ratio: {aspect_ratio:.4f}")
print(f"Warpage: {warpage:.2f}°")
print(f"Skewness: {skewness:.2f}°") 
print(f"Orthogonal Quality: {ortho_quality:.4f}")
```

## Vertex Ordering Convention

All functions expect hexahedral vertices in **standard FEM ordering**:

```
       7----------6
      /|         /|
     / |        / |
    4----------5  |
    |  |       |  |
    |  3-------|--2
    | /        | /
    |/         |/
    0----------1
```

- **Vertices 0-3**: Bottom face (counter-clockwise when viewed from outside)
- **Vertices 4-7**: Top face (counter-clockwise when viewed from outside)
- **Vertical edges**: 0-4, 1-5, 2-6, 3-7

## Quality Metrics Guide

### Aspect Ratio
- **Range**: [1, ∞)
- **Ideal**: 1.0 (cube with equal edge lengths)
- **Acceptable**: ≤ 3.0
- **Poor**: > 10.0

### Warpage
- **Range**: [0°, 180°]  
- **Ideal**: 0° (perfectly planar faces)
- **Acceptable**: ≤ 5°
- **Poor**: > 15°

### Skewness  
- **Range**: [0°, 90°]
- **Ideal**: 0° (all face angles are 90°)
- **Acceptable**: ≤ 10°
- **Poor**: > 30°

### Orthogonal Quality
- **Range**: [0, 1]
- **Ideal**: 1.0 (perfectly orthogonal edges)
- **Acceptable**: ≥ 0.7
- **Poor**: < 0.3

### Scaled Jacobian
- **Range**: [0, 1] 
- **Ideal**: 1.0 (perfect right-handed element)
- **Acceptable**: ≥ 0.5
- **Poor**: < 0.2
- **Invalid**: 0.0 (degenerate/inverted)

## Testing

The package includes comprehensive testing with two options:

### Option A: Full pytest suite (recommended for development)

Install pytest:
```bash
pip install pytest
# OR install with test dependencies:
pip install -e ".[test]"
```

Run comprehensive tests:
```bash
# Run all tests
python -m pytest tests/test_metrics.py -v

# Run tests for specific functions
python -m pytest tests/test_metrics.py::TestAspectRatio -v
python -m pytest tests/test_metrics.py::TestWarpage -v
python -m pytest tests/test_metrics.py::TestSkewness -v
python -m pytest tests/test_metrics.py::TestOrthogonalQuality -v
python -m pytest tests/test_metrics.py::TestJacobian -v
python -m pytest tests/test_metrics.py::TestScaledJacobian -v

# Run with coverage report
python -m pytest tests/test_metrics.py --cov=hex_quality --cov-report=html
```

### Option B: Standalone test runner (no dependencies)

For quick verification without installing pytest:
```bash
# Run essential tests
python run_tests.py

# With verbose output  
python run_tests.py -v
```

### Test Coverage

The test suite includes:
- **Perfect elements**: Unit cubes, scaled cubes (ideal quality metrics)
- **Distorted elements**: Stretched, skewed, warped hexahedra
- **Degenerate cases**: Collapsed edges, collapsed faces, inverted elements
- **Edge cases**: Extreme aspect ratios, near-degenerate elements
- **Input validation**: Various data types, array shapes, error conditions

Each test verifies expected mathematical properties and boundary conditions.

## Examples

### Basic Usage

```python
import hex_quality as hq

# Perfect unit cube
perfect_cube = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
]

# Stretched element (poor aspect ratio)
stretched = [
    [0, 0, 0], [3, 0, 0], [3, 1, 0], [0, 1, 0],  # 3x wider
    [0, 0, 1], [3, 0, 1], [3, 1, 1], [0, 1, 1]
]

print("Perfect Cube:")
print(f"  Aspect Ratio: {hq.aspect_ratio(perfect_cube):.2f}")
print(f"  Orthogonal Quality: {hq.orthogonal_quality(perfect_cube):.2f}")

print("\nStretched Element:")  
print(f"  Aspect Ratio: {hq.aspect_ratio(stretched):.2f}")
print(f"  Orthogonal Quality: {hq.orthogonal_quality(stretched):.2f}")
```

### Jacobian Mapping

```python
import hex_quality as hq

# Reference unit cube
ref_cube = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
]

# Physical element (2x scaled)
phys_cube = [
    [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
    [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]
]

# Jacobian metrics
jac = hq.jacobian(ref_cube, phys_cube)         # 8.0 (volume scale factor)
scaled_jac = hq.scaled_jacobian(ref_cube, phys_cube)  # 1.0 (perfect quality)

print(f"Jacobian: {jac}")
print(f"Scaled Jacobian: {scaled_jac}")
```

### Quality Assessment with Test Verification

```python
import hex_quality as hq
import subprocess

# Run tests to verify everything works (using standalone runner)
result = subprocess.run(["python", "run_tests.py"], capture_output=True, text=True)
print("Test Result:", "PASSED" if result.returncode == 0 else "FAILED")

# Then use the functions normally
vertices = [...]  # your hexahedron vertices
quality_metrics = {
    'aspect_ratio': hq.aspect_ratio(vertices),
    'warpage': hq.warpage(vertices),
    'skewness': hq.skewness(vertices),
    'orthogonal_quality': hq.orthogonal_quality(vertices)
}
print(quality_metrics)
```

## API Reference

### `aspect_ratio(hex_vert)`
Compute aspect ratio (max_edge / min_edge).

### `warpage(hex_vert)`  
Compute maximum face warpage angle in degrees.

### `skewness(hex_vert)`
Compute maximum deviation from 90° face angles.

### `orthogonal_quality(hex_vert)`
Compute orthogonal quality metric [0,1].

### `scaled_jacobian(ref_hex, phys_hex)`
Compute scaled jacobian quality [0,1].

### `jacobian(ref_hex, phys_hex)`  
Compute raw jacobian determinant.

**Parameters:**
- `hex_vert`: array_like, shape (8, 3) - Hexahedron vertex coordinates
- `ref_hex`: array_like, shape (8, 3) - Reference element coordinates  
- `phys_hex`: array_like, shape (8, 3) - Physical element coordinates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Knupp, P. M. (2001). Algebraic mesh quality metrics. *SIAM Journal on Scientific Computing*, 23(1), 193-218.
2. Field, D. A. (2000). Qualitative measures for initial meshes. *International Journal for Numerical Methods in Engineering*, 47(4), 887-906.
3. Pébay, P. P., & Baker, T. J. (2003). Analysis of triangle quality measures. *Mathematics of Computation*, 72(244), 1817-1839.