#!/usr/bin/env python3
"""
Standalone test runner for hex_quality package.

This script runs all tests without requiring pytest installation.
Use this if you don't want to install pytest or just want a quick test.
"""

import sys
import traceback
import numpy as np
from hex_quality.metrics import (
    aspect_ratio,
    warpage,
    skewness,
    orthogonal_quality,
    scaled_jacobian,
    jacobian
)


class TestRunner:
    """Simple test runner that doesn't require pytest."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []

    def assert_equal(self, actual, expected, tolerance=1e-10, message=""):
        """Assert two values are equal within tolerance."""
        if abs(actual - expected) <= tolerance:
            return True
        else:
            raise AssertionError(f"Expected {expected}, got {actual}. {message}")

    def assert_true(self, condition, message=""):
        """Assert condition is true."""
        if not condition:
            raise AssertionError(f"Condition failed. {message}")

    def assert_false(self, condition, message=""):
        """Assert condition is false."""
        if condition:
            raise AssertionError(f"Condition should be false. {message}")

    def run_test(self, test_func, test_name):
        """Run a single test function."""
        try:
            test_func()
            self.passed += 1
            self.test_results.append(f"✓ {test_name}")
            print(f"✓ {test_name}")
        except Exception as e:
            self.failed += 1
            self.test_results.append(f"✗ {test_name}: {str(e)}")
            print(f"✗ {test_name}: {str(e)}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                traceback.print_exc()

    def report(self):
        """Print final test report."""
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {self.passed / total * 100:.1f}%")

        if self.failed > 0:
            print(f"\nFAILED TESTS:")
            for result in self.test_results:
                if result.startswith("✗"):
                    print(f"  {result}")

        return self.failed == 0


# Test fixture geometries
def unit_cube():
    """Perfect unit cube with ideal quality."""
    return [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # top
    ]


def scaled_cube(scale=2.0):
    """Uniformly scaled cube."""
    cube = unit_cube()
    return [[scale * x, scale * y, scale * z] for x, y, z in cube]


def stretched_cube(x_scale=3.0, y_scale=1.0, z_scale=1.0):
    """Non-uniformly stretched cube."""
    cube = unit_cube()
    return [[x_scale * x, y_scale * y, z_scale * z] for x, y, z in cube]


def skewed_hex():
    """Skewed hexahedron."""
    return [
        [0, 0, 0], [1, 0, 0], [1.3, 1, 0], [0.3, 1, 0],  # skewed bottom
        [0, 0, 1], [1, 0, 1], [1.3, 1, 1], [0.3, 1, 1]  # skewed top
    ]


def warped_hex():
    """Hexahedron with warped faces."""
    return [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # flat bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1.2], [0, 1, 0.8]  # warped top
    ]


def collapsed_edge_hex():
    """Degenerate hex with collapsed edge."""
    return [
        [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0],  # edge 0-1 collapsed
        [0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1]
    ]


def inverted_hex():
    """Inverted hexahedron."""
    return [
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],  # reversed bottom face
        [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]
    ]


# Test functions
def test_aspect_ratio_unit_cube():
    """Test aspect ratio of unit cube."""
    runner.assert_equal(aspect_ratio(unit_cube()), 1.0, message="Unit cube should have AR=1.0")


def test_aspect_ratio_scaled_cube():
    """Test aspect ratio of scaled cube."""
    runner.assert_equal(aspect_ratio(scaled_cube(5.0)), 1.0, message="Scaled cube should maintain AR=1.0")


def test_aspect_ratio_stretched_cube():
    """Test aspect ratio of stretched cube."""
    stretch_factor = 3.0
    ar = aspect_ratio(stretched_cube(stretch_factor, 1.0, 1.0))
    runner.assert_equal(ar, stretch_factor, message=f"Stretched cube should have AR={stretch_factor}")


def test_aspect_ratio_collapsed_edge():
    """Test aspect ratio with collapsed edge."""
    ar = aspect_ratio(collapsed_edge_hex())
    runner.assert_true(ar == float('inf'), message="Collapsed edge should return infinity")


def test_warpage_unit_cube():
    """Test warpage of unit cube."""
    runner.assert_equal(warpage(unit_cube()), 0.0, message="Unit cube should have zero warpage")


def test_warpage_scaled_cube():
    """Test warpage of scaled cube."""
    runner.assert_equal(warpage(scaled_cube(3.0)), 0.0, message="Scaled cube should maintain zero warpage")


def test_warpage_warped_hex():
    """Test warpage of warped hexahedron."""
    warp = warpage(warped_hex())
    runner.assert_true(warp > 0, message="Warped hex should have positive warpage")
    runner.assert_true(warp < 180, message="Warpage should be less than 180°")


def test_skewness_unit_cube():
    """Test skewness of unit cube."""
    runner.assert_equal(skewness(unit_cube()), 0.0, message="Unit cube should have zero skewness")


def test_skewness_skewed_hex():
    """Test skewness of skewed hexahedron."""
    skew = skewness(skewed_hex())
    runner.assert_true(skew > 0, message="Skewed hex should have positive skewness")
    runner.assert_true(skew <= 90, message="Skewness should be <= 90°")


def test_orthogonal_quality_unit_cube():
    """Test orthogonal quality of unit cube."""
    runner.assert_equal(orthogonal_quality(unit_cube()), 1.0, message="Unit cube should have OQ=1.0")


def test_orthogonal_quality_scaled_cube():
    """Test orthogonal quality of scaled cube."""
    runner.assert_equal(orthogonal_quality(scaled_cube(4.0)), 1.0, message="Scaled cube should maintain OQ=1.0")


def test_orthogonal_quality_skewed_hex():
    """Test orthogonal quality of skewed hexahedron."""
    oq = orthogonal_quality(skewed_hex())
    runner.assert_true(0 <= oq < 1.0, message="Skewed hex should have 0 <= OQ < 1.0")


def test_orthogonal_quality_bounds():
    """Test orthogonal quality bounds."""
    test_cases = [unit_cube(), scaled_cube(0.5), stretched_cube(2, 1, 1), skewed_hex(), warped_hex()]
    for i, vertices in enumerate(test_cases):
        oq = orthogonal_quality(vertices)
        runner.assert_true(0.0 <= oq <= 1.0, message=f"OQ out of bounds for case {i}: {oq}")


def test_jacobian_identity_mapping():
    """Test jacobian for identity mapping."""
    ref = unit_cube()
    phys = unit_cube()
    jac = jacobian(ref, phys)
    runner.assert_equal(jac, 1.0, message="Identity mapping should have jacobian=1.0")


def test_jacobian_scaled_mapping():
    """Test jacobian for scaled mapping."""
    ref = unit_cube()
    scale = 2.0
    phys = scaled_cube(scale)
    jac = jacobian(ref, phys)
    expected = scale ** 3
    runner.assert_equal(jac, expected, message=f"Scaled mapping should have jacobian={expected}")


def test_jacobian_inverted_mapping():
    """Test jacobian for inverted mapping."""
    ref = unit_cube()
    phys = inverted_hex()
    jac = jacobian(ref, phys)
    runner.assert_true(jac < 0, message="Inverted mapping should have negative jacobian")


def test_scaled_jacobian_identity_mapping():
    """Test scaled jacobian for identity mapping."""
    ref = unit_cube()
    phys = unit_cube()
    sj = scaled_jacobian(ref, phys)
    runner.assert_equal(sj, 1.0, message="Identity mapping should have scaled jacobian=1.0")


def test_scaled_jacobian_scaled_mapping():
    """Test scaled jacobian for uniform scaling."""
    ref = unit_cube()
    phys = scaled_cube(3.0)
    sj = scaled_jacobian(ref, phys)
    runner.assert_equal(sj, 1.0, message="Uniform scaling should maintain scaled jacobian=1.0")


def test_scaled_jacobian_inverted_mapping():
    """Test scaled jacobian for inverted mapping."""
    ref = unit_cube()
    phys = inverted_hex()
    sj = scaled_jacobian(ref, phys)
    runner.assert_equal(sj, 0.0, message="Inverted mapping should have scaled jacobian=0.0")


def test_scaled_jacobian_bounds():
    """Test scaled jacobian bounds."""
    ref = unit_cube()
    test_cases = [unit_cube(), scaled_cube(0.5), stretched_cube(2, 1, 1), skewed_hex(), warped_hex()]
    for i, phys in enumerate(test_cases):
        sj = scaled_jacobian(ref, phys)
        runner.assert_true(0.0 <= sj <= 1.0, message=f"Scaled jacobian out of bounds for case {i}: {sj}")


def test_original_jacobian_example():
    """Test the original jacobian example from main.py."""
    ref = unit_cube()
    phys = scaled_cube(2.0)

    jac = jacobian(ref, phys)
    sj = scaled_jacobian(ref, phys)

    runner.assert_equal(jac, 8.0, message="Original example should have jacobian=8.0")
    runner.assert_equal(sj, 1.0, message="Original example should have scaled jacobian=1.0")


def test_numpy_array_input():
    """Test that numpy arrays work as input."""
    vertices_list = unit_cube()
    vertices_array = np.array(vertices_list)

    ar1 = aspect_ratio(vertices_list)
    ar2 = aspect_ratio(vertices_array)
    runner.assert_equal(ar1, ar2, message="List and numpy array should give same result")


def test_extreme_aspect_ratios():
    """Test very extreme aspect ratios."""
    vertices = stretched_cube(1000.0, 1.0, 1.0)
    ar = aspect_ratio(vertices)
    runner.assert_equal(ar, 1000.0, tolerance=1e-8, message="Extreme stretch should give correct AR")


# Main test execution
def main():
    """Run all tests."""
    global runner
    runner = TestRunner()

    print("Running hex_quality package tests...")
    print("=" * 60)

    # List of all test functions
    test_functions = [
        (test_aspect_ratio_unit_cube, "aspect_ratio: unit cube"),
        (test_aspect_ratio_scaled_cube, "aspect_ratio: scaled cube"),
        (test_aspect_ratio_stretched_cube, "aspect_ratio: stretched cube"),
        (test_aspect_ratio_collapsed_edge, "aspect_ratio: collapsed edge"),
        (test_warpage_unit_cube, "warpage: unit cube"),
        (test_warpage_scaled_cube, "warpage: scaled cube"),
        (test_warpage_warped_hex, "warpage: warped hex"),
        (test_skewness_unit_cube, "skewness: unit cube"),
        (test_skewness_skewed_hex, "skewness: skewed hex"),
        (test_orthogonal_quality_unit_cube, "orthogonal_quality: unit cube"),
        (test_orthogonal_quality_scaled_cube, "orthogonal_quality: scaled cube"),
        (test_orthogonal_quality_skewed_hex, "orthogonal_quality: skewed hex"),
        (test_orthogonal_quality_bounds, "orthogonal_quality: bounds check"),
        (test_jacobian_identity_mapping, "jacobian: identity mapping"),
        (test_jacobian_scaled_mapping, "jacobian: scaled mapping"),
        (test_jacobian_inverted_mapping, "jacobian: inverted mapping"),
        (test_scaled_jacobian_identity_mapping, "scaled_jacobian: identity mapping"),
        (test_scaled_jacobian_scaled_mapping, "scaled_jacobian: scaled mapping"),
        (test_scaled_jacobian_inverted_mapping, "scaled_jacobian: inverted mapping"),
        (test_scaled_jacobian_bounds, "scaled_jacobian: bounds check"),
        (test_original_jacobian_example, "regression: original jacobian example"),
        (test_numpy_array_input, "input: numpy array compatibility"),
        (test_extreme_aspect_ratios, "edge case: extreme aspect ratios"),
    ]

    # Run all tests
    for test_func, test_name in test_functions:
        runner.run_test(test_func, test_name)

    # Print summary
    success = runner.report()

    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
