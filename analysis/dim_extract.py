# Ethan Doughty
# dim_extract.py
"""Dimension extraction, struct field updates, and argument unwrapping."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from ir import Expr, Var, Const, End, BinOp, IndexArg, IndexExpr
from analysis.end_helpers import _binop_contains_end, _eval_end_arithmetic

import analysis.diagnostics as diag
from runtime.env import Env
from runtime.shapes import Shape, Dim, SymDim, add_dim, mul_dim, sub_dim

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


def expr_to_dim_with_end(expr: Expr, env: Env, end_dim: Dim) -> Dim:
    """Convert an expression containing End to a dimension value.

    Like expr_to_dim_ir but substitutes End nodes with end_dim, enabling
    symbolic extent computation for range indexing (e.g., 1:end, 1:end-1).

    Args:
        expr: Expression that may contain End nodes
        env: Current environment (for variable dim aliases)
        end_dim: Dimension value to substitute for End (int, SymDim, or None)

    Returns:
        Dimension value, or None if not determinable
    """
    if isinstance(expr, End):
        return end_dim
    if isinstance(expr, Const):
        v = expr.value
        return int(v) if float(v).is_integer() else None
    if isinstance(expr, Var):
        # Check for dimension alias first (propagates caller's dim name)
        if expr.name in env.dim_aliases:
            return env.dim_aliases[expr.name]
        return SymDim.var(expr.name)
    if isinstance(expr, BinOp):
        # Recursively extract dimensions from left and right operands
        left = expr_to_dim_with_end(expr.left, env, end_dim)
        right = expr_to_dim_with_end(expr.right, env, end_dim)
        if left is None or right is None:
            return None
        if expr.op == "+":
            return add_dim(left, right)
        if expr.op == "-":
            return sub_dim(left, right)
        if expr.op == "*":
            return mul_dim(left, right)
        return None  # unsupported operator (/, .*, etc.)
    # Neg, other nodes: not expected in index expressions
    return None


def expr_to_dim_ir_with_end(expr: Expr, env: Env, container_dim: Optional[int]) -> Optional[int]:
    """Convert an expression containing End to a concrete integer if possible.

    This is used for range endpoints like end-1:end where we need to resolve
    end to a concrete value.

    Args:
        expr: Expression to evaluate (may contain End)
        env: Current environment
        container_dim: Container dimension to use for End resolution (e.g., num_cols)

    Returns:
        Concrete integer if resolvable, None otherwise
    """
    if isinstance(expr, Const):
        v = expr.value
        if float(v).is_integer():
            return int(v)
        return None
    if isinstance(expr, End):
        # End keyword resolves to container dimension
        return container_dim
    if isinstance(expr, BinOp) and _binop_contains_end(expr):
        # Arithmetic with End (e.g., end-1, end/2)
        if container_dim is not None:
            return _eval_end_arithmetic(expr, container_dim)
        return None
    # For other expressions (Var, etc.), fall back to expr_to_dim_ir
    dim = expr_to_dim_ir(expr, env)
    if isinstance(dim, int):
        return dim
    return None


def expr_to_dim_ir(expr: Expr, env: Env) -> Dim:
    """Convert an expression to a dimension value if possible.

    Args:
        expr: Expression to convert
        env: Current environment

    Returns:
        Integer dimension, symbolic name, or None if not determinable
    """
    if isinstance(expr, Const):
        v = expr.value
        if float(v).is_integer():
            return int(v)
        return None
    if isinstance(expr, End):
        # End keyword: can't convert to dimension without container context
        return None
    if isinstance(expr, Var):
        # Check for dimension alias first (propagates caller's dim name)
        if expr.name in env.dim_aliases:
            return env.dim_aliases[expr.name]
        return SymDim.var(expr.name)
    if isinstance(expr, BinOp):
        # Recursively extract dimensions from left and right operands
        left_dim = expr_to_dim_ir(expr.left, env)
        right_dim = expr_to_dim_ir(expr.right, env)

        # If either operand cannot be extracted, return None
        if left_dim is None or right_dim is None:
            return None

        # Handle supported operators
        if expr.op == "+":
            return add_dim(left_dim, right_dim)
        elif expr.op == "-":
            # Subtraction via negation
            return add_dim(left_dim, mul_dim(-1, right_dim))
        elif expr.op == "*":
            return mul_dim(left_dim, right_dim)
        else:
            # Unsupported operators: .*, ./, /, :, etc.
            return None
    return None


def unwrap_arg(arg: IndexArg) -> Expr:
    """Extract the inner Expr from an IndexArg.

    Used by builtin handlers to access argument expressions when Apply args
    are stored as IndexArg (which can include Colon, Range, or IndexExpr).

    Args:
        arg: IndexArg to unwrap

    Returns:
        The inner Expr if arg is IndexExpr

    Raises:
        ValueError: If arg is Colon or Range (cannot be unwrapped to Expr)
    """
    if isinstance(arg, IndexExpr):
        return arg.expr
    raise ValueError(f"Cannot unwrap {type(arg).__name__} to Expr")


def _update_struct_field(
    base_shape: Shape,
    fields: List[str],
    value_shape: Shape,
    line: int,
    warnings: List['Diagnostic']
) -> Shape:
    """Update a nested struct field with a new value.

    Args:
        base_shape: Current shape of the base variable (may be bottom if unbound)
        fields: Chain of field names (e.g., ["a", "b"] for s.a.b)
        value_shape: Shape to assign to the field
        line: Source line number (for warnings)
        warnings: List to append warnings to

    Returns:
        Updated struct shape with the field set to value_shape
    """
    # If base is bottom (unbound variable), create fresh struct from chain
    if base_shape.is_bottom():
        # Build nested struct from innermost field outward
        result = value_shape
        for field in reversed(fields):
            result = Shape.struct({field: result})
        return result

    # If base is not a struct, warn and treat as fresh struct
    if not base_shape.is_struct():
        warnings.append(diag.warn_field_access_non_struct(line, base_shape))
        # Still create struct (best-effort recovery)
        result = value_shape
        for field in reversed(fields):
            result = Shape.struct({field: result})
        return result

    # Base is struct: walk the chain and update
    if len(fields) == 1:
        # Simple case: s.field = value
        updated_fields = dict(base_shape.fields_dict)
        updated_fields[fields[0]] = value_shape
        return Shape.struct(updated_fields)
    else:
        # Nested case: s.a.b = value
        # Recursively update s.a, then update s
        outer_field = fields[0]
        inner_fields = fields[1:]

        # Get current shape of outer field (or bottom if missing)
        outer_field_shape = base_shape.fields_dict.get(outer_field, Shape.bottom())

        # Recursively update inner fields
        updated_inner = _update_struct_field(outer_field_shape, inner_fields, value_shape, line, warnings)

        # Update outer struct with new inner value
        updated_fields = dict(base_shape.fields_dict)
        updated_fields[outer_field] = updated_inner
        return Shape.struct(updated_fields)
