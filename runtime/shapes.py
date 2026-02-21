# Ethan Doughty
# shapes.py
"""Shape abstract domain for MATLAB static analysis.

Defines the Shape type and dimension operations used throughout the analyzer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

from runtime.symdim import SymDim


# A dimension can be:
# - an int (e.g., 3, 4) for concrete dimensions
# - a SymDim for symbolic dimension expressions
# - None for completely unknown dimensions
Dim = Union[int, SymDim, None]


class Shape:
    """Abstract base for all shape kinds.

    Subclasses are frozen dataclasses (one per kind). This base class provides:
    - Class-level attribute defaults so all consumer code (``._elements``,
      ``._lambda_ids``, ``.rows``, ``.cols``) works without modification.
    - Default ``is_*()`` predicates (all return False).
    - Static factory methods that return the appropriate subclass.
    - ``fields_dict`` property.
    """

    # Class-level defaults so attribute access never crashes on any subclass
    rows: Optional[Dim] = None
    cols: Optional[Dim] = None
    _fields: tuple = ()
    _lambda_ids: Optional[frozenset] = None
    _elements: Optional[tuple] = None

    # Constructors

    @staticmethod
    def scalar() -> "ScalarShape":
        """Create a scalar shape."""
        return ScalarShape()

    @staticmethod
    def matrix(rows: Dim, cols: Dim) -> "MatrixShape":
        """Create a matrix shape with given dimensions."""
        return MatrixShape(rows, cols)

    @staticmethod
    def unknown() -> "UnknownShape":
        """Create an unknown shape (for error cases)."""
        return UnknownShape()

    @staticmethod
    def bottom() -> "BottomShape":
        """Create a bottom shape (no information / unbound variable)."""
        return BottomShape()

    @staticmethod
    def string() -> "StringShape":
        """Create a string shape (char array literal)."""
        return StringShape()

    @staticmethod
    def struct(fields: dict, open: bool = False) -> "StructShape":
        """Create a struct shape with given fields.

        Args:
            fields: Dict mapping field names to shapes
            open: If True, struct may have additional unknown fields (open lattice element)

        Returns:
            Struct shape with fields stored as sorted tuple for hashability
        """
        return StructShape(_fields=tuple(sorted(fields.items())), _open=open)

    @staticmethod
    def cell(rows: Dim, cols: Dim, elements: Optional[dict] = None) -> "CellShape":
        """Create a cell array shape with optional per-element shapes.

        Args:
            rows: Number of rows
            cols: Number of columns
            elements: Optional dict mapping linear indices to shapes

        Returns:
            Cell shape with _elements as sorted tuple of (idx, shape) pairs
        """
        if elements is None:
            elem_tuple = None
        else:
            elem_tuple = tuple(sorted(elements.items()))
        return CellShape(rows, cols, elem_tuple)

    @staticmethod
    def function_handle(lambda_ids=None) -> "FunctionHandleShape":
        """Create a function handle shape (anonymous or named).

        Args:
            lambda_ids: Optional frozenset of lambda/handle IDs for precise analysis
        """
        return FunctionHandleShape(lambda_ids)

    @property
    def fields_dict(self) -> dict:
        """Get fields as a dict (for struct shapes only).

        Returns:
            Dict of field names to shapes, or empty dict if not a struct
        """
        return dict(self._fields) if self._fields else {}

    # Predicates (all return False on the base class)

    def is_scalar(self) -> bool:
        """Check if this is a scalar shape."""
        return False

    def is_matrix(self) -> bool:
        """Check if this is a matrix shape."""
        return False

    def is_unknown(self) -> bool:
        """Check if this is an unknown shape."""
        return False

    def is_bottom(self) -> bool:
        """Check if this is a bottom shape."""
        return False

    def is_string(self) -> bool:
        """Check if this is a string shape."""
        return False

    def is_struct(self) -> bool:
        """Check if this is a struct shape."""
        return False

    def is_cell(self) -> bool:
        """Check if this is a cell array shape."""
        return False

    def is_function_handle(self) -> bool:
        """Check if this is a function handle shape."""
        return False

    def is_numeric(self) -> bool:
        """Check if this is a numeric type (scalar, matrix, or string).

        Strings are numeric because MATLAB treats char arrays as numeric values.
        """
        return False

    def is_empty_matrix(self) -> bool:
        """Check if this is a 0x0 matrix (MATLAB's universal empty initializer [])."""
        return False


# ---------------------------------------------------------------------------
# Frozen-dataclass subclasses (one per shape kind)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarShape(Shape):
    """A scalar (single numeric value)."""
    kind: str = "scalar"  # class-level attribute for backward compat string access

    def is_scalar(self) -> bool:
        return True

    def is_numeric(self) -> bool:
        return True

    def __str__(self) -> str:
        return "scalar"


@dataclass(frozen=True)
class MatrixShape(Shape):
    """A 2-D matrix with symbolic or concrete dimensions."""
    rows: Dim = None
    cols: Dim = None
    kind: str = "matrix"

    def is_matrix(self) -> bool:
        return True

    def is_numeric(self) -> bool:
        return True

    def is_empty_matrix(self) -> bool:
        return self.rows == 0 and self.cols == 0

    def __str__(self) -> str:
        return f"matrix[{_dim_str(self.rows)} x {_dim_str(self.cols)}]"


@dataclass(frozen=True)
class StringShape(Shape):
    """A string (char array)."""
    kind: str = "string"

    def is_string(self) -> bool:
        return True

    def is_numeric(self) -> bool:
        # MATLAB treats char arrays as numeric values
        return True

    def __str__(self) -> str:
        return "string"


@dataclass(frozen=True)
class StructShape(Shape):
    """A struct with named fields."""
    _fields: tuple = ()
    _open: bool = False
    kind: str = "struct"

    @property
    def fields_dict(self) -> dict:
        return dict(self._fields) if self._fields else {}

    def is_struct(self) -> bool:
        return True

    def __str__(self) -> str:
        # Filter out bottom fields (internal-only, shouldn't appear in user output)
        field_strs = [f"{name}: {shape}" for name, shape in self._fields if not shape.is_bottom()]
        if self._open:
            field_strs.append("...")
        return "struct{" + ", ".join(field_strs) + "}"


@dataclass(frozen=True)
class FunctionHandleShape(Shape):
    """A function handle (anonymous function or named handle)."""
    _lambda_ids: Optional[frozenset] = None
    kind: str = "function_handle"

    def is_function_handle(self) -> bool:
        return True

    def __str__(self) -> str:
        return "function_handle"


@dataclass(frozen=True)
class CellShape(Shape):
    """A cell array with optional per-element shape tracking."""
    rows: Dim = None
    cols: Dim = None
    _elements: Optional[tuple] = None
    kind: str = "cell"

    def is_cell(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"cell[{_dim_str(self.rows)} x {_dim_str(self.cols)}]"


@dataclass(frozen=True)
class UnknownShape(Shape):
    """Unknown/indeterminate shape (top of the lattice)."""
    kind: str = "unknown"

    def is_unknown(self) -> bool:
        return True

    def __str__(self) -> str:
        return "unknown"


@dataclass(frozen=True)
class BottomShape(Shape):
    """Bottom shape — no information / unbound variable (lattice identity)."""
    kind: str = "bottom"

    def is_bottom(self) -> bool:
        return True

    def __str__(self) -> str:
        return "bottom"  # Should never appear in user-visible output


# Dimension helpers

def _dim_str(d: Dim) -> str:
    """Format a dimension for display in shape strings."""
    if d is None:
        return "None"
    if isinstance(d, int):
        return str(d)
    if isinstance(d, SymDim):
        s = str(d)
        # Bare constant or bare variable: no parens
        cv = d.const_value()
        if cv is not None:
            return str(cv)
        # Check if it's a bare variable: single term, coeff 1, single var with exp 1
        if len(d._terms) == 1:
            mono, coeff = d._terms[0]
            if coeff == 1 and len(mono) == 1 and mono[0][1] == 1:
                return mono[0][0]  # Just the variable name
        return f"({s})"
    return str(d)


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


def _to_symdim(d: Dim) -> SymDim:
    """Convert a Dim to SymDim (helper for arithmetic)."""
    if isinstance(d, int):
        return SymDim.const(d)
    if isinstance(d, SymDim):
        return d
    raise TypeError(f"Cannot convert {type(d)} to SymDim")


def dims_definitely_conflict(a: Dim, b: Dim) -> bool:
    """Return True if we can prove the dimensions are different"""
    if a is None or b is None:
        return False
    if a == b:
        return False
    # Both concrete ints
    if isinstance(a, int) and isinstance(b, int):
        return True  # a != b already checked above
    # Check if difference is a nonzero constant
    try:
        sa, sb = _to_symdim(a), _to_symdim(b)
        diff = sa - sb
        cv = diff.const_value()
        if cv is not None and cv != 0:
            return True
    except TypeError:
        pass
    return False


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
    sa, sb = _to_symdim(a), _to_symdim(b)
    result = sa + sb
    cv = result.const_value()
    return cv if cv is not None else result


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
    sa, sb = _to_symdim(a), _to_symdim(b)
    result = sa * sb
    cv = result.const_value()
    return cv if cv is not None else result


def sub_dim(a: Dim, b: Dim) -> Dim:
    """Subtract dimension b from a: a - b.

    Args:
        a: First dimension
        b: Second dimension to subtract

    Returns:
        Difference of dimensions
    """
    if b is None:
        return None
    return add_dim(a, mul_dim(-1, b))


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


def _traverse_shapes(s1: Shape, s2: Shape, dim_op) -> Shape:
    """Generic shape lattice traversal parameterized by a dimension operator.

    Implements the shared structure of join_shape and widen_shape:
    bottom-as-identity, unknown-as-top, then 6 same-kind match arms
    (scalar, matrix, string, function_handle, struct, cell), then
    cross-kind fallback to unknown.

    Args:
        s1: First shape
        s2: Second shape
        dim_op: Callable(Dim, Dim) -> Dim applied to matrix/cell dimensions.
                Use join_dim for join_shape, widen_dim for widen_shape.

    Returns:
        Result shape according to dim_op and the shared lattice rules.
    """
    # Bottom is identity (no information from unbound variable)
    if s1.is_bottom():
        return s2
    if s2.is_bottom():
        return s1

    # Unknown is absorbing top (error/indeterminate propagates)
    if s1.is_unknown() or s2.is_unknown():
        return Shape.unknown()

    if s1.is_scalar() and s2.is_scalar():
        return Shape.scalar()

    if s1.is_matrix() and s2.is_matrix():
        return Shape.matrix(dim_op(s1.rows, s2.rows), dim_op(s1.cols, s2.cols))

    if s1.is_string() and s2.is_string():
        return Shape.string()

    if s1.is_function_handle() and s2.is_function_handle():
        # Union lambda_ids; None (opaque) is absorbing
        ids1, ids2 = s1._lambda_ids, s2._lambda_ids
        merged_ids = None if (ids1 is None or ids2 is None) else ids1 | ids2
        return Shape.function_handle(lambda_ids=merged_ids)

    if s1.is_struct() and s2.is_struct():
        # Open structs propagate: if either is open, result is open
        result_open = s1._open or s2._open
        all_fields = set(s1.fields_dict.keys()) | set(s2.fields_dict.keys())
        merged_fields = {}
        for field_name in all_fields:
            # Missing field in open struct → unknown; in closed struct → bottom
            default1 = Shape.unknown() if s1._open else Shape.bottom()
            default2 = Shape.unknown() if s2._open else Shape.bottom()
            f1 = s1.fields_dict.get(field_name, default1)
            f2 = s2.fields_dict.get(field_name, default2)
            merged_fields[field_name] = _traverse_shapes(f1, f2, dim_op)
        return Shape.struct(merged_fields, open=result_open)

    if s1.is_cell() and s2.is_cell():
        merged_rows = dim_op(s1.rows, s2.rows)
        merged_cols = dim_op(s1.cols, s2.cols)
        # None is absorbing: if either side has no element tracking, result has none
        if s1._elements is None or s2._elements is None:
            merged_elements = None
        else:
            dict1, dict2 = dict(s1._elements), dict(s2._elements)
            all_indices = set(dict1.keys()) | set(dict2.keys())
            elem_dict = {}
            for idx in all_indices:
                e1 = dict1.get(idx, Shape.bottom())
                e2 = dict2.get(idx, Shape.bottom())
                elem_dict[idx] = _traverse_shapes(e1, e2, dim_op)
            # Remove bottom elements (internal-only, don't store)
            elem_dict = {idx: s for idx, s in elem_dict.items() if not s.is_bottom()}
            merged_elements = elem_dict if elem_dict else None
        return Shape.cell(merged_rows, merged_cols, elements=merged_elements)

    # Different kinds → unknown
    return Shape.unknown()


def widen_shape(old: Shape, new: Shape) -> Shape:
    """Widen shape pointwise. Bottom is identity, unknown is absorbing top.

    Used in fixpoint loop analysis for both widening and post-loop join.
    Delegates to _traverse_shapes with widen_dim as the dimension operator.

    Args:
        old: Shape from previous iteration or pre-loop environment
        new: Shape from current iteration

    Returns:
        Widened shape (stable dims preserved, conflicting dims widened to None)
    """
    return _traverse_shapes(old, new, widen_dim)


# Shape lattice join

def join_shape(s1: Shape, s2: Shape) -> Shape:
    """Pointwise join of two shapes. Bottom is identity, unknown is absorbing top.

    Delegates to _traverse_shapes with join_dim as the dimension operator.

    Args:
        s1: First shape
        s2: Second shape

    Returns:
        Joined shape
    """
    return _traverse_shapes(s1, s2, join_dim)


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
