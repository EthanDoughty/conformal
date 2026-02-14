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

    @staticmethod
    def bottom() -> "Shape":
        """Create a bottom shape (no information / unbound variable)."""
        return Shape(kind="bottom")

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

    def is_bottom(self) -> bool:
        """Check if this is a bottom shape."""
        return self.kind == "bottom"

    # Pretty print / debug

    def __str__(self) -> str:
        if self.kind == "scalar":
            return "scalar"
        if self.kind == "matrix":
            return f"matrix[{self.rows} x {self.cols}]"
        if self.kind == "bottom":
            return "bottom"  # Should never appear in user-visible output
        return "unknown"

    def __repr__(self) -> str:
        return f"Shape(kind={self.kind!r}, rows={self.rows!r}, cols={self.cols!r})"


# Dimension helpers

def join_dim(a: Dim, b: Dim) -> Dim:
    """Join two dimensions in the lattice.

    None represents "unknown dimension" (top for dims).
    When joining different concrete values or any value with None, result is None.
    """
    if a == b:
        return a
    # None is absorbing (unknown dimension stays unknown)
    if a is None or b is None:
        return None
    # Different concrete values → unknown
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
    if isinstance(b, (int, float)) and b < 0:
        return f"({a}-{-b})"
    if isinstance(b, str) and b.startswith("-"):
        return f"({a}-{b[1:]})"
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


# Widening operators for fixpoint loop analysis

def widen_dim(old: Dim, new: Dim) -> Dim:
    """Widen dimension: stable dims preserved, conflicting dims -> None.

    Used in fixpoint loop analysis to accelerate convergence.
    If old == new, the dimension is stable; otherwise widen to None.

    Args:
        old: Dimension from previous iteration
        new: Dimension from current iteration

    Returns:
        old if dimensions match, None otherwise
    """
    if old == new:
        return old
    return None


def widen_shape(old: Shape, new: Shape) -> Shape:
    """Widen shape pointwise. Bottom is identity, unknown is absorbing top.

    Used in fixpoint loop analysis for both widening and post-loop join.

    Lattice semantics:
    - bottom is identity: widen(bottom, s) = s, widen(s, bottom) = s
    - unknown is top: widen(unknown, s) = unknown, widen(s, unknown) = unknown
    - Pointwise widening for matrices: stable dims preserved, conflicts -> None

    Args:
        old: Shape from previous iteration or pre-loop environment
        new: Shape from current iteration

    Returns:
        Widened shape with bottom-as-identity, unknown-as-top, pointwise widen_dim
    """
    # Bottom is identity (no information from unbound variable)
    if old.is_bottom():
        return new
    if new.is_bottom():
        return old

    # Unknown is absorbing top (error/indeterminate propagates)
    if old.is_unknown() or new.is_unknown():
        return Shape.unknown()

    # Both known kinds: pointwise widening
    if old.is_scalar() and new.is_scalar():
        return Shape.scalar()
    if old.is_matrix() and new.is_matrix():
        return Shape.matrix(
            widen_dim(old.rows, new.rows),
            widen_dim(old.cols, new.cols),
        )
    return Shape.unknown()  # different kinds (scalar vs matrix) -> unknown


# Shape lattice join

def join_shape(s1: Shape, s2: Shape) -> Shape:
    """Pointwise join of two shapes. Bottom is identity, unknown is absorbing top.

    Lattice semantics:
    - bottom is identity: join(bottom, s) = s, join(s, bottom) = s
    - unknown is top: join(unknown, s) = unknown, join(s, unknown) = unknown
    - Pointwise join_dim for matrices

    Args:
        s1: First shape
        s2: Second shape

    Returns:
        Joined shape
    """
    # Bottom is identity
    if s1.is_bottom():
        return s2
    if s2.is_bottom():
        return s1

    # Unknown is absorbing top
    if s1.is_unknown() or s2.is_unknown():
        return Shape.unknown()

    # Both known kinds: pointwise join
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