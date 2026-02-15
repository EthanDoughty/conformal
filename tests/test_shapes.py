"""Unit tests for SymDim rational coefficients."""
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from runtime.shapes import SymDim, Shape, add_dim, mul_dim
from fractions import Fraction


class TestRationalDimensions(unittest.TestCase):
    def test_division_creates_rational(self):
        """Test that division creates rational coefficients."""
        n = SymDim.var("n")
        half_n = n / SymDim.const(2)
        # Check coefficient is Fraction(1, 2)
        self.assertEqual(len(half_n._terms), 1)
        mono, coeff = half_n._terms[0]
        self.assertEqual(coeff, Fraction(1, 2))

    def test_rational_addition(self):
        """Test that rational coefficients add correctly."""
        n = SymDim.var("n")
        half_n = n / SymDim.const(2)
        result = half_n + half_n
        self.assertEqual(result, n)

    def test_rational_display_simple(self):
        """Test display format for simple rational (n/2)."""
        n = SymDim.var("n")
        half_n = n / SymDim.const(2)
        s = str(half_n)
        # Should be "n/2" (special case for 1/d * single var)
        self.assertEqual(s, "n/2")

    def test_rational_display_complex(self):
        """Test display format for complex rational (3/2)*n."""
        n = SymDim.var("n")
        three_half_n = (n * SymDim.const(3)) / SymDim.const(2)
        s = str(three_half_n)
        # Should be "(3/2)*n"
        self.assertEqual(s, "(3/2)*n")

    def test_rational_in_shape(self):
        """Test rational dimensions in Shape display."""
        n = SymDim.var("n")
        half_n = n / SymDim.const(2)
        shape = Shape.matrix(half_n, 1)
        s = str(shape)
        # Should contain "n/2"
        self.assertIn("n/2", s)

    def test_division_by_zero_raises(self):
        """Test that division by zero raises ValueError."""
        n = SymDim.var("n")
        with self.assertRaises(ValueError):
            n / SymDim.const(0)

    def test_division_non_constant_raises(self):
        """Test that division by non-constant raises ValueError."""
        n = SymDim.var("n")
        m = SymDim.var("m")
        with self.assertRaises(ValueError):
            n / m

    def test_fraction_normalization(self):
        """Test that Fraction normalizes automatically (2/4 -> 1/2)."""
        n = SymDim.var("n")
        # (n * 2) / 4 should normalize to n/2
        two_n = n * SymDim.const(2)
        result = two_n / SymDim.const(4)
        # Check coefficient is Fraction(1, 2), not Fraction(2, 4)
        mono, coeff = result._terms[0]
        self.assertEqual(coeff, Fraction(1, 2))


if __name__ == "__main__":
    unittest.main()
