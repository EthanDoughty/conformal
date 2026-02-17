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


@dataclass(frozen=True)
class Shape:
    """Abstract shape for MATLAB values.

    Represents one of: scalar, matrix[rows x cols], string, struct, function_handle, unknown, or bottom.
    Matrix dimensions can be concrete integers, symbolic names, or None.
    """
    kind: str
    rows: Optional[Dim] = None
    cols: Optional[Dim] = None
    _fields: tuple = ()  # For struct shapes: tuple of (field_name, Shape) pairs (sorted)
    _lambda_ids: Optional[frozenset] = None  # For function_handle shapes: set of lambda/handle IDs
    _elements: Optional[tuple] = None  # For cell shapes: tuple of (linear_index, Shape) pairs (sorted)

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

    @staticmethod
    def string() -> "Shape":
        """Create a string shape (char array literal)."""
        return Shape(kind="string")

    @staticmethod
    def struct(fields: dict) -> "Shape":
        """Create a struct shape with given fields.

        Args:
            fields: Dict mapping field names to shapes

        Returns:
            Struct shape with fields stored as sorted tuple for hashability
        """
        return Shape(kind="struct", _fields=tuple(sorted(fields.items())))

    @staticmethod
    def cell(rows: Dim, cols: Dim, elements: Optional[dict] = None) -> "Shape":
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
        return Shape(kind="cell", rows=rows, cols=cols, _elements=elem_tuple)

    @staticmethod
    def function_handle(lambda_ids=None) -> "Shape":
        """Create a function handle shape (anonymous or named).

        Args:
            lambda_ids: Optional frozenset of lambda/handle IDs for precise analysis
        """
        return Shape(kind="function_handle", _lambda_ids=lambda_ids)

    @property
    def fields_dict(self) -> dict:
        """Get fields as a dict (for struct shapes only).

        Returns:
            Dict of field names to shapes, or empty dict if not a struct
        """
        return dict(self._fields) if self.kind == "struct" else {}

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

    def is_string(self) -> bool:
        """Check if this is a string shape."""
        return self.kind == "string"

    def is_struct(self) -> bool:
        """Check if this is a struct shape."""
        return self.kind == "struct"

    def is_cell(self) -> bool:
        """Check if this is a cell array shape."""
        return self.kind == "cell"

    def is_function_handle(self) -> bool:
        """Check if this is a function handle shape."""
        return self.kind == "function_handle"

    def is_numeric(self) -> bool:
        """Check if this is a numeric type (scalar, matrix, or string).

        Strings are numeric because MATLAB treats char arrays as numeric values.
        """
        return self.kind in ("scalar", "matrix", "string")

    # Pretty print / debug

    def __str__(self) -> str:
        if self.kind == "scalar":
            return "scalar"
        if self.kind == "matrix":
            return f"matrix[{_dim_str(self.rows)} x {_dim_str(self.cols)}]"
        if self.kind == "string":
            return "string"
        if self.kind == "struct":
            # Format: struct{x: scalar, y: matrix[3 x 1]}
            # Filter out bottom fields (internal-only, shouldn't appear in user output)
            field_strs = [f"{name}: {shape}" for name, shape in self._fields if not shape.is_bottom()]
            return "struct{" + ", ".join(field_strs) + "}"
        if self.kind == "cell":
            return f"cell[{_dim_str(self.rows)} x {_dim_str(self.cols)}]"
        if self.kind == "function_handle":
            return "function_handle"
        if self.kind == "bottom":
            return "bottom"  # Should never appear in user-visible output
        return "unknown"

    def __repr__(self) -> str:
        return f"Shape(kind={self.kind!r}, rows={self.rows!r}, cols={self.cols!r})"


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


def widen_shape(old: Shape, new: Shape) -> Shape:
    """Widen shape pointwise. Bottom is identity, unknown is absorbing top.

    Used in fixpoint loop analysis for both widening and post-loop join.

    Lattice semantics:
    - bottom is identity: widen(bottom, s) = s, widen(s, bottom) = s
    - unknown is top: widen(unknown, s) = unknown, widen(s, unknown) = unknown
    - Same kind → same kind (pointwise for matrices)
    - Different kinds → unknown

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
    if old.is_string() and new.is_string():
        return Shape.string()

    if old.is_function_handle() and new.is_function_handle():
        # Union lambda_ids; None (opaque) is absorbing
        ids1 = old._lambda_ids
        ids2 = new._lambda_ids
        if ids1 is None or ids2 is None:
            merged_ids = None
        else:
            merged_ids = ids1 | ids2
        return Shape.function_handle(lambda_ids=merged_ids)

    if old.is_struct() and new.is_struct():
        # Union-with-bottom widen: union of all field names, missing fields get bottom
        all_fields = set(old.fields_dict.keys()) | set(new.fields_dict.keys())
        widened_fields = {}
        for field_name in all_fields:
            f_old = old.fields_dict.get(field_name, Shape.bottom())
            f_new = new.fields_dict.get(field_name, Shape.bottom())
            widened_fields[field_name] = widen_shape(f_old, f_new)
        return Shape.struct(widened_fields)

    if old.is_cell() and new.is_cell():
        # Widen dimensions
        widened_rows = widen_dim(old.rows, new.rows)
        widened_cols = widen_dim(old.cols, new.cols)

        # Widen elements: None is absorbing (if either side has no tracking, result has no tracking)
        if old._elements is None or new._elements is None:
            widened_elements = None
        else:
            # Merge element dicts pointwise (union keys, widen values)
            old_dict = dict(old._elements)
            new_dict = dict(new._elements)
            all_indices = set(old_dict.keys()) | set(new_dict.keys())
            widened_elem_dict = {}
            for idx in all_indices:
                e_old = old_dict.get(idx, Shape.bottom())
                e_new = new_dict.get(idx, Shape.bottom())
                widened_elem_dict[idx] = widen_shape(e_old, e_new)
            # Remove bottom elements (internal-only, don't store)
            widened_elem_dict = {idx: s for idx, s in widened_elem_dict.items() if not s.is_bottom()}
            widened_elements = widened_elem_dict if widened_elem_dict else None

        return Shape.cell(widened_rows, widened_cols, elements=widened_elements)

    # Different kinds → unknown
    return Shape.unknown()


# Shape lattice join

def join_shape(s1: Shape, s2: Shape) -> Shape:
    """Pointwise join of two shapes. Bottom is identity, unknown is absorbing top.

    Lattice semantics:
    - bottom is identity: join(bottom, s) = s, join(s, bottom) = s
    - unknown is top: join(unknown, s) = unknown, join(s, unknown) = unknown
    - Same kind → same kind (pointwise for matrices)
    - Different kinds → unknown

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

    if s1.is_string() and s2.is_string():
        return Shape.string()

    if s1.is_function_handle() and s2.is_function_handle():
        # Union lambda_ids; None (opaque) is absorbing
        ids1 = s1._lambda_ids
        ids2 = s2._lambda_ids
        if ids1 is None or ids2 is None:
            merged_ids = None
        else:
            merged_ids = ids1 | ids2
        return Shape.function_handle(lambda_ids=merged_ids)

    if s1.is_struct() and s2.is_struct():
        # Union-with-bottom join: union of all field names, missing fields get bottom
        all_fields = set(s1.fields_dict.keys()) | set(s2.fields_dict.keys())
        joined_fields = {}
        for field_name in all_fields:
            f1 = s1.fields_dict.get(field_name, Shape.bottom())
            f2 = s2.fields_dict.get(field_name, Shape.bottom())
            joined_fields[field_name] = join_shape(f1, f2)
        return Shape.struct(joined_fields)

    if s1.is_cell() and s2.is_cell():
        # Join dimensions
        r = join_dim(s1.rows, s2.rows)
        c = join_dim(s1.cols, s2.cols)

        # Join elements: None is absorbing (if either side has no tracking, result has no tracking)
        if s1._elements is None or s2._elements is None:
            joined_elements = None
        else:
            # Merge element dicts pointwise (union keys, join values)
            dict1 = dict(s1._elements)
            dict2 = dict(s2._elements)
            all_indices = set(dict1.keys()) | set(dict2.keys())
            joined_elem_dict = {}
            for idx in all_indices:
                e1 = dict1.get(idx, Shape.bottom())
                e2 = dict2.get(idx, Shape.bottom())
                joined_elem_dict[idx] = join_shape(e1, e2)
            # Remove bottom elements (internal-only, don't store)
            joined_elem_dict = {idx: s for idx, s in joined_elem_dict.items() if not s.is_bottom()}
            joined_elements = joined_elem_dict if joined_elem_dict else None

        return Shape.cell(r, c, elements=joined_elements)

    # Different kinds → unknown
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