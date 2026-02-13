# Ethan Doughty
# shapes.py
"""Shape abstract domain for Mini-MATLAB static analysis.

Defines the Shape type and dimension operations used throughout the analyzer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

# A dimension can be:
# - an int (e.g., 3, 4) for concrete dimensions
# - a symbolic name (e.g., "n", "m") for unknown but tracked dimensions
# - None for completely unknown dimensions
Dim = Union[int, str, None]


@dataclass(frozen=True)
class Shape:
    """Abstract shape for MATLAB values.

    Represents one of: scalar, matrix[rows x cols], or unknown.
    Matrix dimensions can be concrete integers, symbolic names, or None.
    """
    kind: str
    rows: Optional[Dim] = None
    cols: Optional[Dim] = None

    # Constructors

    @staticmethod
    def scalar() -> "Shape":
        """Create a scalar shape."""
        return Shape(kind="scalar")

    @staticmethod
    def matrix(rows: Dim, cols: Dim) -> "Shape":
        """Create a matrix shape with given dimensions."""
        return Shape(kind="matrix", rows=rows, cols=cols)

    @staticmethod
    def unknown() -> "Shape":
        """Create an unknown shape (for error cases)."""
        return Shape(kind="unknown")

    # Predicates

    def is_scalar(self) -> bool:
        """Check if this is a scalar shape."""
        return self.kind == "scalar"

    def is_matrix(self) -> bool:
        """Check if this is a matrix shape."""
        return self.kind == "matrix"

    def is_unknown(self) -> bool:
        """Check if this is an unknown shape."""
        return self.kind == "unknown"

    # Pretty print / debug

    def __str__(self) -> str:
        if self.kind == "scalar":
            return "scalar"
        if self.kind == "matrix":
            return f"matrix[{self.rows} x {self.cols}]"
        return "unknown"

    def __repr__(self) -> str:
        return f"Shape(kind={self.kind!r}, rows={self.rows!r}, cols={self.cols!r})"


# Dimension helpers

def join_dim(a: Dim, b: Dim) -> Dim:
    """Join two dimensions in the lattice."""
    if a == b:
        return a
    if a is None:
        return b
    if b is None:
        return a
    return None


def dims_definitely_conflict(a: Dim, b: Dim) -> bool:
    """Return True if we can prove the dimensions are different"""
    if a is None or b is None:
        return False
    return a != b

def add_dim(a: Dim, b: Dim) -> Dim:
    """Add two dimensions symbolically.

    Args:
        a: First dimension
        b: Second dimension

    Returns:
        Sum of dimensions (concrete if both are ints, symbolic otherwise)
    """
    if a is None or b is None:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return f"({a}+{b})"


def mul_dim(a: Dim, b: Dim) -> Dim:
    """Multiply two dimensions symbolically.

    Args:
        a: First dimension
        b: Second dimension

    Returns:
        Product of dimensions (concrete if both are ints, symbolic otherwise)
        Short-circuits: mul_dim(0, x) → 0, mul_dim(1, x) → x (both directions)
    """
    # Short-circuit: 0 * x = 0 (and x * 0 = 0)
    if a == 0 or b == 0:
        return 0
    # Short-circuit: 1 * x = x (and x * 1 = x)
    if a == 1:
        return b
    if b == 1:
        return a
    # Unknown dimension in either position
    if a is None or b is None:
        return None
    # Both concrete: multiply
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    # Symbolic: create symbolic product
    return f"({a}*{b})"


def sum_dims(dimensions: list[Dim]) -> Dim:
    """Sum a list of dimensions.

    Args:
        dimensions: List of dimensions to sum

    Returns:
        Total dimension (0 if empty list)
    """
    if not dimensions:
        return 0
    total = dimensions[0]
    for dim in dimensions[1:]:
        total = add_dim(total, dim)
    return total

# Shape lattice join

def join_shape(s1: Shape, s2: Shape) -> Shape:
    """Pointwise join of two shapes"""
    if s1.is_unknown():
        return s2
    if s2.is_unknown():
        return s1

    # both known kinds
    if s1.is_scalar() and s2.is_scalar():
        return Shape.scalar()

    if s1.is_matrix() and s2.is_matrix():
        r = join_dim(s1.rows, s2.rows)
        c = join_dim(s1.cols, s2.cols)
        return Shape.matrix(r, c)

    return Shape.unknown()


# Convenience functions for common patterns

def shape_of_zeros(rows: Dim, cols: Dim) -> Shape:
    """Shape for zeros(m, n) or ones(m, n)"""
    return Shape.matrix(rows, cols)


def shape_of_ones(rows: Dim, cols: Dim) -> Shape:
    """Shape for ones(m, n)."""
    return Shape.matrix(rows, cols)


def shape_of_colon(start: Dim, end: Dim) -> Shape:
    """Shape for 1:n style vectors.

    Args:
        start: Start value (currently unused, assumed to be 1)
        end: End value (becomes column dimension)

    Returns:
        Row vector shape (1 x end)
    """
    return Shape.matrix(1, end)