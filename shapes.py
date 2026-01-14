# Ethan Doughty
# shapes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

# A dimension can be:
# an int  (3, 4)
# a symbolic name ("n", "m")
# None for "unknown"
Dim = Union[int, str, None]


@dataclass(frozen=True)
class Shape:
    """Abstract shape for MATLAB values"""
    kind: str
    rows: Optional[Dim] = None
    cols: Optional[Dim] = None

    # Constructors

    @staticmethod
    def scalar() -> "Shape":
        return Shape(kind="scalar")

    @staticmethod
    def matrix(rows: Dim, cols: Dim) -> "Shape":
        return Shape(kind="matrix", rows=rows, cols=cols)

    @staticmethod
    def unknown() -> "Shape":
        return Shape(kind="unknown")

    # Predicates

    def is_scalar(self) -> bool:
        return self.kind == "scalar"

    def is_matrix(self) -> bool:
        return self.kind == "matrix"

    def is_unknown(self) -> bool:
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
    """Symbolic/int addition for dimensions."""
    if a is None or b is None:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return f"({a}+{b})"

def sum_dims(ds: list[Dim]) -> Dim:
    if not ds:
        return 0
    acc = ds[0]
    for d in ds[1:]:
        acc = add_dim(acc, d)
    return acc

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
    return Shape.matrix(rows, cols)


def shape_of_colon(start: Dim, end: Dim) -> Shape:
    """Shape for 1:n style vectors"""
    return Shape.matrix(1, end)