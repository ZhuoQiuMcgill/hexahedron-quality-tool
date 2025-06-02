"""
Comprehensive test suite for hex_quality package.

This module contains unit tests for all six quality metric functions,
testing both normal cases and edge cases to ensure robustness.
"""

import pytest
import numpy as np
from hex_quality.metrics import (
    aspect_ratio,
    warpage,
    skewness,
    orthogonal_quality,
    scaled_jacobian,
    jacobian
)


class TestHexGeometries:
    """Test fixture geometries for various hexahedral elements."""

    @staticmethod
    def unit_cube():
        """Perfect unit cube with ideal quality."""
        return [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # top
        ]

    @staticmethod
    def scaled_cube(scale=2.0):
        """Uniformly scaled cube (should maintain quality)."""
        cube = TestHexGeometries.unit_cube()
        return [[scale * x, scale * y, scale * z] for x, y, z in cube]

    @staticmethod
    def stretched_cube(x_scale=3.0, y_scale=1.0, z_scale=1.0):
        """Non-uniformly stretched cube (poor aspect ratio)."""
        cube = TestHexGeometries.unit_cube()
        return [[x_scale * x, y_scale * y, z_scale * z] for x, y, z in cube]

    @staticmethod
    def skewed_hex():
        """Skewed hexahedron (poor orthogonal quality)."""
        return [
            [0, 0, 0], [1, 0, 0], [1.3, 1, 0], [0.3, 1, 0],  # skewed bottom
            [0, 0, 1], [1, 0, 1], [1.3, 1, 1], [0.3, 1, 1]  # skewed top
        ]

    @staticmethod
    def warped_hex():
        """Hexahedron with warped faces (non-planar)."""
        return [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # flat bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1.2], [0, 1, 0.8]  # warped top
        ]

    @staticmethod
    def collapsed_edge_hex():
        """Degenerate hex with collapsed edge."""
        return [
            [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0],  # edge 0-1 collapsed
            [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1]
        ]

    @staticmethod
    def inverted_hex():
        """Inverted hexahedron (negative jacobian)."""
        return [
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],  # reversed bottom face
            [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]
        ]


class TestAspectRatio:
    """Test cases for aspect_ratio function."""

    def test_unit_cube_aspect_ratio(self):
        """Unit cube should have aspect ratio of 1.0."""
        vertices = TestHexGeometries.unit_cube()
        ar = aspect_ratio(vertices)
        assert abs(ar - 1.0) < 1e-10, f"Expected 1.0, got {ar}"

    def test_scaled_cube_aspect_ratio(self):
        """Uniformly scaled cube should maintain aspect ratio of 1.0."""
        vertices = TestHexGeometries.scaled_cube(5.0)
        ar = aspect_ratio(vertices)
        assert abs(ar - 1.0) < 1e-10, f"Expected 1.0, got {ar}"

    def test_stretched_cube_aspect_ratio(self):
        """Stretched cube should have aspect ratio equal to stretch factor."""
        stretch_factor = 3.0
        vertices = TestHexGeometries.stretched_cube(stretch_factor, 1.0, 1.0)
        ar = aspect_ratio(vertices)
        assert abs(ar - stretch_factor) < 1e-10, f"Expected {stretch_factor}, got {ar}"

    def test_collapsed_edge_aspect_ratio(self):
        """Collapsed edge should return infinity."""
        vertices = TestHexGeometries.collapsed_edge_hex()
        ar = aspect_ratio(vertices)
        assert ar == float('inf'), f"Expected inf, got {ar}"

    @pytest.mark.parametrize("stretch_x,stretch_y,stretch_z", [
        (2.0, 1.0, 1.0),
        (1.0, 3.0, 1.0),
        (1.0, 1.0, 4.0),
        (2.0, 3.0, 1.0)
    ])
    def test_various_stretches(self, stretch_x, stretch_y, stretch_z):
        """Test aspect ratio for various stretch configurations."""
        vertices = TestHexGeometries.stretched_cube(stretch_x, stretch_y, stretch_z)
        ar = aspect_ratio(vertices)
        expected = max(stretch_x, stretch_y, stretch_z) / min(stretch_x, stretch_y, stretch_z)
        assert abs(ar - expected) < 1e-10, f"Expected {expected}, got {ar}"


class TestWarpage:
    """Test cases for warpage function."""

    def test_unit_cube_warpage(self):
        """Unit cube should have zero warpage."""
        vertices = TestHexGeometries.unit_cube()
        warp = warpage(vertices)
        assert abs(warp) < 1e-10, f"Expected 0.0, got {warp}"

    def test_scaled_cube_warpage(self):
        """Scaled cube should maintain zero warpage."""
        vertices = TestHexGeometries.scaled_cube(3.0)
        warp = warpage(vertices)
        assert abs(warp) < 1e-10, f"Expected 0.0, got {warp}"

    def test_warped_hex_warpage(self):
        """Warped hexahedron should have non-zero warpage."""
        vertices = TestHexGeometries.warped_hex()
        warp = warpage(vertices)
        assert warp > 0, f"Expected positive warpage, got {warp}"
        assert warp < 180, f"Warpage should be less than 180°, got {warp}"

    def test_collapsed_face_warpage(self):
        """Collapsed face should return infinity."""
        vertices = TestHexGeometries.collapsed_edge_hex()
        warp = warpage(vertices)
        assert warp == float('inf'), f"Expected inf, got {warp}"


class TestSkewness:
    """Test cases for skewness function."""

    def test_unit_cube_skewness(self):
        """Unit cube should have zero skewness."""
        vertices = TestHexGeometries.unit_cube()
        skew = skewness(vertices)
        assert abs(skew) < 1e-10, f"Expected 0.0, got {skew}"

    def test_scaled_cube_skewness(self):
        """Scaled cube should maintain zero skewness."""
        vertices = TestHexGeometries.scaled_cube(2.5)
        skew = skewness(vertices)
        assert abs(skew) < 1e-10, f"Expected 0.0, got {skew}"

    def test_skewed_hex_skewness(self):
        """Skewed hexahedron should have non-zero skewness."""
        vertices = TestHexGeometries.skewed_hex()
        skew = skewness(vertices)
        assert skew > 0, f"Expected positive skewness, got {skew}"
        assert skew <= 90, f"Skewness should be <= 90°, got {skew}"

    def test_collapsed_edge_skewness(self):
        """Collapsed edge should return infinity."""
        vertices = TestHexGeometries.collapsed_edge_hex()
        skew = skewness(vertices)
        assert skew == float('inf'), f"Expected inf, got {skew}"


class TestOrthogonalQuality:
    """Test cases for orthogonal_quality function."""

    def test_unit_cube_orthogonal_quality(self):
        """Unit cube should have perfect orthogonal quality (1.0)."""
        vertices = TestHexGeometries.unit_cube()
        oq = orthogonal_quality(vertices)
        assert abs(oq - 1.0) < 1e-10, f"Expected 1.0, got {oq}"

    def test_scaled_cube_orthogonal_quality(self):
        """Scaled cube should maintain perfect orthogonal quality."""
        vertices = TestHexGeometries.scaled_cube(4.0)
        oq = orthogonal_quality(vertices)
        assert abs(oq - 1.0) < 1e-10, f"Expected 1.0, got {oq}"

    def test_skewed_hex_orthogonal_quality(self):
        """Skewed hexahedron should have reduced orthogonal quality."""
        vertices = TestHexGeometries.skewed_hex()
        oq = orthogonal_quality(vertices)
        assert 0 <= oq < 1.0, f"Expected 0 <= oq < 1.0, got {oq}"

    def test_collapsed_edge_orthogonal_quality(self):
        """Collapsed edge should return zero quality."""
        vertices = TestHexGeometries.collapsed_edge_hex()
        oq = orthogonal_quality(vertices)
        assert oq == 0.0, f"Expected 0.0, got {oq}"

    def test_orthogonal_quality_bounds(self):
        """Orthogonal quality should always be in [0, 1]."""
        test_cases = [
            TestHexGeometries.unit_cube(),
            TestHexGeometries.scaled_cube(0.5),
            TestHexGeometries.stretched_cube(2, 1, 1),
            TestHexGeometries.skewed_hex(),
            TestHexGeometries.warped_hex()
        ]

        for vertices in test_cases:
            oq = orthogonal_quality(vertices)
            assert 0.0 <= oq <= 1.0, f"OQ out of bounds [0,1]: {oq}"


class TestJacobian:
    """Test cases for jacobian function."""

    def test_identity_mapping_jacobian(self):
        """Identity mapping should have jacobian of 1.0."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.unit_cube()
        jac = jacobian(ref, phys)
        assert abs(jac - 1.0) < 1e-10, f"Expected 1.0, got {jac}"

    def test_scaled_mapping_jacobian(self):
        """Uniform scaling should have jacobian equal to scale³."""
        ref = TestHexGeometries.unit_cube()
        scale = 2.0
        phys = TestHexGeometries.scaled_cube(scale)
        jac = jacobian(ref, phys)
        expected = scale ** 3  # Volume scaling factor
        assert abs(jac - expected) < 1e-10, f"Expected {expected}, got {jac}"

    def test_inverted_mapping_jacobian(self):
        """Inverted mapping should have negative jacobian."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.inverted_hex()
        jac = jacobian(ref, phys)
        assert jac < 0, f"Expected negative jacobian, got {jac}"

    @pytest.mark.parametrize("scale", [0.5, 1.5, 3.0, 10.0])
    def test_various_scale_jacobians(self, scale):
        """Test jacobian for various uniform scaling factors."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.scaled_cube(scale)
        jac = jacobian(ref, phys)
        expected = scale ** 3
        assert abs(jac - expected) < 1e-10, f"Expected {expected}, got {jac}"


class TestScaledJacobian:
    """Test cases for scaled_jacobian function."""

    def test_identity_mapping_scaled_jacobian(self):
        """Identity mapping should have scaled jacobian of 1.0."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.unit_cube()
        sj = scaled_jacobian(ref, phys)
        assert abs(sj - 1.0) < 1e-10, f"Expected 1.0, got {sj}"

    def test_scaled_mapping_scaled_jacobian(self):
        """Uniform scaling should maintain scaled jacobian of 1.0."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.scaled_cube(3.0)
        sj = scaled_jacobian(ref, phys)
        assert abs(sj - 1.0) < 1e-10, f"Expected 1.0, got {sj}"

    def test_inverted_mapping_scaled_jacobian(self):
        """Inverted mapping should have scaled jacobian of 0.0."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.inverted_hex()
        sj = scaled_jacobian(ref, phys)
        assert abs(sj) < 1e-10, f"Expected 0.0, got {sj}"

    def test_degenerate_mapping_scaled_jacobian(self):
        """Degenerate mapping should have scaled jacobian of 0.0."""
        ref = TestHexGeometries.unit_cube()
        phys = TestHexGeometries.collapsed_edge_hex()
        sj = scaled_jacobian(ref, phys)
        assert abs(sj) < 1e-10, f"Expected 0.0, got {sj}"

    def test_scaled_jacobian_bounds(self):
        """Scaled jacobian should always be in [0, 1]."""
        ref = TestHexGeometries.unit_cube()
        test_cases = [
            TestHexGeometries.unit_cube(),
            TestHexGeometries.scaled_cube(0.5),
            TestHexGeometries.stretched_cube(2, 1, 1),
            TestHexGeometries.skewed_hex(),
            TestHexGeometries.warped_hex()
        ]

        for phys in test_cases:
            sj = scaled_jacobian(ref, phys)
            assert 0.0 <= sj <= 1.0, f"Scaled jacobian out of bounds [0,1]: {sj}"

    def test_input_validation(self):
        """Test input validation for scaled_jacobian."""
        ref = TestHexGeometries.unit_cube()

        # Test wrong shape
        with pytest.raises(ValueError, match="must be an \\(8, 3\\) array"):
            scaled_jacobian([[0, 0, 0]], ref)  # Wrong number of vertices

        with pytest.raises(ValueError, match="must be an \\(8, 3\\) array"):
            scaled_jacobian(ref, [[0, 0]])  # Wrong dimension


class TestInputFormats:
    """Test various input formats and edge cases."""

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        vertices_list = TestHexGeometries.unit_cube()
        vertices_array = np.array(vertices_list)

        # All functions should work with numpy arrays
        ar1 = aspect_ratio(vertices_list)
        ar2 = aspect_ratio(vertices_array)
        assert abs(ar1 - ar2) < 1e-10

        warp1 = warpage(vertices_list)
        warp2 = warpage(vertices_array)
        assert abs(warp1 - warp2) < 1e-10

    def test_flattened_input(self):
        """Test that functions handle various input shapes."""
        vertices = TestHexGeometries.unit_cube()
        vertices_flat = np.array(vertices).reshape(24)  # Flattened

        # Functions should reshape automatically
        ar = aspect_ratio(vertices_flat)
        assert abs(ar - 1.0) < 1e-10

    def test_different_dtypes(self):
        """Test that functions work with different numeric types."""
        vertices_int = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        vertices_float = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]

        ar_int = aspect_ratio(vertices_int)
        ar_float = aspect_ratio(vertices_float)
        assert abs(ar_int - ar_float) < 1e-10


class TestRegressionCases:
    """Test cases that reproduce specific scenarios or bug fixes."""

    def test_original_jacobian_example(self):
        """Test the original jacobian example from main.py."""
        # Reference unit cube
        ref = TestHexGeometries.unit_cube()

        # Perfect 2× scaled cube (right-handed)
        phys = TestHexGeometries.scaled_cube(2.0)

        jac = jacobian(ref, phys)
        sj = scaled_jacobian(ref, phys)

        assert abs(jac - 8.0) < 1e-10, f"Expected 8.0, got {jac}"
        assert abs(sj - 1.0) < 1e-10, f"Expected 1.0, got {sj}"

    def test_extreme_aspect_ratios(self):
        """Test very extreme aspect ratios."""
        vertices = TestHexGeometries.stretched_cube(1000.0, 1.0, 1.0)
        ar = aspect_ratio(vertices)
        assert abs(ar - 1000.0) < 1e-8, f"Expected 1000.0, got {ar}"

    def test_near_degenerate_cases(self):
        """Test cases that are nearly degenerate but not quite."""
        epsilon = 1e-12
        vertices = [
            [0, 0, 0], [epsilon, 0, 0], [1, 1, 0], [0, 1, 0],  # Very small edge
            [0, 0, 1], [epsilon, 0, 1], [1, 1, 1], [0, 1, 1]
        ]

        ar = aspect_ratio(vertices)
        assert ar > 1e10, f"Expected very large aspect ratio, got {ar}"
        assert ar != float('inf'), f"Should not be infinite for non-zero edge"


# Utility functions for running specific test groups
def test_all_quality_metrics():
    """Run all quality metric tests."""
    pytest.main([__file__, "-v"])


def test_basic_functionality():
    """Run basic functionality tests only."""
    pytest.main([__file__ + "::TestAspectRatio::test_unit_cube_aspect_ratio",
                 __file__ + "::TestWarpage::test_unit_cube_warpage",
                 __file__ + "::TestSkewness::test_unit_cube_skewness",
                 __file__ + "::TestOrthogonalQuality::test_unit_cube_orthogonal_quality",
                 __file__ + "::TestJacobian::test_identity_mapping_jacobian",
                 __file__ + "::TestScaledJacobian::test_identity_mapping_scaled_jacobian",
                 "-v"])


if __name__ == "__main__":
    # Run all tests when executed directly
    test_all_quality_metrics()
