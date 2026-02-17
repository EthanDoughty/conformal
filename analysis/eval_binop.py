# Ethan Doughty
# eval_binop.py
"""Binary operation shape inference."""

from __future__ import annotations
from typing import List, TYPE_CHECKING

from ir import Expr

import analysis.diagnostics as diag
from runtime.shapes import Shape, join_dim, dims_definitely_conflict
from analysis.constraints import record_constraint

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


def eval_binop_ir(
    op: str,
    left: Shape,
    right: Shape,
    warnings: List['Diagnostic'],
    left_expr: Expr,
    right_expr: Expr,
    line: int,
    ctx,
    env
) -> Shape:
    """Evaluate a binary operation and infer result shape.

    Args:
        op: Operator string (+, -, *, .*, etc.)
        left: Left operand shape
        right: Right operand shape
        warnings: List to append warnings to
        left_expr: Left expression (for error messages)
        right_expr: Right expression (for error messages)
        line: Source line number
        ctx: Analysis context
        env: Current environment

    Returns:
        Result shape of the operation
    """
    if op in {"==", "~=", "<", "<=", ">", ">="}:
        if (left.is_matrix() and right.is_scalar()) or (left.is_scalar() and right.is_matrix()):
            warnings.append(diag.warn_suspicious_comparison_matrix_scalar(line, op, left_expr, right_expr, left, right)
            )
        elif left.is_matrix() and right.is_matrix():
            warnings.append(diag.warn_matrix_to_matrix_comparison(line, op, left_expr, right_expr, left, right)
            )
        return Shape.scalar()

    if op in {"&&", "||"}:
        if left.is_matrix() or right.is_matrix():
            warnings.append(diag.warn_logical_op_non_scalar(line, op, left_expr, right_expr, left, right)
            )
        return Shape.scalar()

    if op == ":":
        return Shape.matrix(1, None)

    # String arithmetic: string + string = numeric row vector (MATLAB behavior)
    if op == "+" and left.is_string() and right.is_string():
        return Shape.matrix(1, None)  # char + char = numeric row vector, length unknown

    # String + non-string: warning + unknown
    if op in {"+", "-", "*", ".*", "/", "./"} and (left.is_string() or right.is_string()):
        if not (left.is_string() and right.is_string() and op == "+"):
            warnings.append(diag.warn_string_arithmetic(line, op, left, right))
            return Shape.unknown()

    # Type guard: non-numeric types cannot participate in arithmetic
    if not (left.is_unknown() or right.is_unknown()):
        if not left.is_numeric() or not right.is_numeric():
            warnings.append(diag.warn_arithmetic_type_mismatch(line, op, left_expr, right_expr, left, right))
            return Shape.unknown()

    if left.is_scalar() and not right.is_scalar():
        return right
    if right.is_scalar() and not left.is_scalar():
        return left

    if op in {"+", "-", ".*", "./", "/"}:
        if left.is_unknown() or right.is_unknown():
            return Shape.unknown()

        if left.is_scalar() and right.is_scalar():
            # Division-by-zero check
            if op in {"/", "./"}:
                from analysis.eval_expr import _get_expr_interval
                from analysis.intervals import interval_is_exactly_zero

                divisor_iv = _get_expr_interval(right_expr, env, ctx)
                if interval_is_exactly_zero(divisor_iv):
                    warnings.append(diag.warn_division_by_zero(line, left_expr, right_expr))

            return Shape.scalar()

        if left.is_matrix() and right.is_matrix():
            # Record elementwise constraints before checking conflicts
            record_constraint(ctx, env, left.rows, right.rows, line)
            record_constraint(ctx, env, left.cols, right.cols, line)

            r_conflict = dims_definitely_conflict(left.rows, right.rows)
            c_conflict = dims_definitely_conflict(left.cols, right.cols)
            if r_conflict or c_conflict:
                warnings.append(diag.warn_elementwise_mismatch(line, op, left_expr, right_expr, left, right)
                )
                return Shape.unknown()

            rows = join_dim(left.rows, right.rows)
            cols = join_dim(left.cols, right.cols)
            return Shape.matrix(rows, cols)

        return Shape.unknown()

    if op == "*":
        if left.is_scalar() and right.is_scalar():
            return Shape.scalar()

        if left.is_scalar() and right.is_matrix():
            return right
        if right.is_scalar() and left.is_matrix():
            return left

        if left.is_matrix() and right.is_matrix():
            # Record matrix multiplication inner dimension constraint
            record_constraint(ctx, env, left.cols, right.rows, line)

            if dims_definitely_conflict(left.cols, right.rows):
                suggest = (
                    not dims_definitely_conflict(left.rows, right.rows)
                    and not dims_definitely_conflict(left.cols, right.cols)
                )
                warnings.append(diag.warn_matmul_mismatch(line, left_expr, right_expr, left, right, suggest)
                )
                return Shape.unknown()

            return Shape.matrix(left.rows, right.cols)

        return Shape.unknown()

    return Shape.unknown()
