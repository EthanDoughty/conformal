# Ethan Doughty
# analysis_ir.py
"""IR-based static shape analyzer for Mini-MATLAB.

This module performs shape inference on the typed IR AST, tracking matrix
dimensions and detecting dimension mismatches at compile time.
"""

from __future__ import annotations
from typing import List, Tuple

from ir import (
    Program, Stmt,
    Assign, ExprStmt, While, For, If, OpaqueStmt,
    Expr, Var, Const, MatrixLit, Call, Transpose, Neg, Index, BinOp,
    IndexArg, Colon, Range, IndexExpr,
)

import analysis.diagnostics as diag
from runtime.env import Env
from runtime.shapes import Shape, Dim, shape_of_zeros, shape_of_ones, join_dim
from analysis.analysis_core import shapes_definitely_incompatible
from analysis.matrix_literals import infer_matrix_literal_shape, as_matrix_shape, dims_definitely_conflict


def analyze_program_ir(program: Program) -> Tuple[Env, List[str]]:
    """Analyze a complete Mini-MATLAB program for shape consistency.

    Args:
        program: IR program to analyze

    Returns:
        Tuple of (final environment, list of warning messages)
    """
    env = Env()
    warnings: List[str] = []
    for stmt in program.body:
        analyze_stmt_ir(stmt, env, warnings)
    return env, warnings


def analyze_stmt_ir(stmt: Stmt, env: Env, warnings: List[str]) -> Env:
    """Analyze a statement and update environment with inferred shapes.

    Args:
        stmt: Statement to analyze
        env: Current environment (modified in place)
        warnings: List to append warnings to

    Returns:
        Updated environment
    """
    if isinstance(stmt, Assign):
        new_shape = eval_expr_ir(stmt.expr, env, warnings)
        old_shape = env.get(stmt.name)

        if stmt.name in env.bindings and shapes_definitely_incompatible(old_shape, new_shape):
            warnings.append(diag.warn_reassign_incompatible(stmt.line, stmt.name, new_shape, old_shape))

        env.set(stmt.name, new_shape)
        return env

    if isinstance(stmt, ExprStmt):
        _ = eval_expr_ir(stmt.expr, env, warnings)
        return env

    if isinstance(stmt, While):
        _ = eval_expr_ir(stmt.cond, env, warnings)
        for s in stmt.body:
            analyze_stmt_ir(s, env, warnings)
        return env

    if isinstance(stmt, For):
        # Naive single-pass analysis (no fixed-point iteration)
        for s in stmt.body:
            analyze_stmt_ir(s, env, warnings)
        return env

    if isinstance(stmt, If):
        _ = eval_expr_ir(stmt.cond, env, warnings)

        then_env = env.copy()
        else_env = env.copy()

        for s in stmt.then_body:
            analyze_stmt_ir(s, then_env, warnings)
        for s in stmt.else_body:
            analyze_stmt_ir(s, else_env, warnings)

        from runtime.env import join_env
        merged = join_env(then_env, else_env)
        env.bindings = merged.bindings
        return env

    if isinstance(stmt, OpaqueStmt):
        # Emit warning for unsupported statement
        warnings.append(diag.warn_unsupported_stmt(stmt.line, stmt.raw, stmt.targets))
        # Havoc all target variables (set to unknown)
        for target_name in stmt.targets:
            env.set(target_name, Shape.unknown())
        return env

    return env


def eval_expr_ir(expr: Expr, env: Env, warnings: List[str]) -> Shape:
    """Evaluate an expression to infer its shape.

    Args:
        expr: Expression to evaluate
        env: Current environment with variable shapes
        warnings: List to append warnings to

    Returns:
        Inferred shape of the expression
    """
    # Variables / constants
    if isinstance(expr, Var):
        return env.get(expr.name)

    if isinstance(expr, Const):
        return Shape.scalar()

    if isinstance(expr, MatrixLit):
        shape_rows = [
            [as_matrix_shape(eval_expr_ir(e, env, warnings)) for e in row]
            for row in expr.rows
        ]
        return infer_matrix_literal_shape(shape_rows, expr.line, warnings)

    if isinstance(expr, Call):
        if isinstance(expr.func, Var):
            fname = expr.func.name
            if fname in {"zeros", "ones"} and len(expr.args) == 2:
                r_dim = expr_to_dim_ir(expr.args[0], env)
                c_dim = expr_to_dim_ir(expr.args[1], env)
                return shape_of_zeros(r_dim, c_dim) if fname == "zeros" else shape_of_ones(r_dim, c_dim)
            if fname == "size":
                if len(expr.args) == 1:
                    # size(A) returns a 1x2 row vector [rows, cols]
                    eval_expr_ir(expr.args[0], env, warnings)
                    return Shape.matrix(1, 2)
                elif len(expr.args) == 2:
                    # size(A, dim) returns a scalar
                    eval_expr_ir(expr.args[0], env, warnings)
                    return Shape.scalar()
            if fname == "isscalar" and len(expr.args) == 1:
                # isscalar(x) always returns a logical scalar
                eval_expr_ir(expr.args[0], env, warnings)
                return Shape.scalar()

        return Shape.scalar()

    if isinstance(expr, Transpose):
        inner = eval_expr_ir(expr.operand, env, warnings)
        if inner.is_matrix():
            return Shape.matrix(inner.cols, inner.rows)
        return inner

    if isinstance(expr, Neg):
        return eval_expr_ir(expr.operand, env, warnings)

    if isinstance(expr, Index):
        line = expr.line
        base_shape = eval_expr_ir(expr.base, env, warnings)

        if base_shape.is_unknown():
            return Shape.unknown()

        if base_shape.is_scalar():
            warnings.append(diag.warn_indexing_scalar(line, expr))
            return Shape.unknown()

        if base_shape.is_matrix():
            m = base_shape.rows
            n = base_shape.cols
            args = expr.args

            if len(args) == 1:
                return Shape.scalar()

            if len(args) == 2:
                a1, a2 = args

                r_extent = index_arg_to_extent_ir(a1, env, warnings, line)
                c_extent = index_arg_to_extent_ir(a2, env, warnings, line)

                # Strict: if extent unknown due to invalid indexing, return unknown
                # (Colon/Range are allowed to have unknown extent)
                def is_allowed_unknown(a: IndexArg) -> bool:
                    return isinstance(a, (Colon, Range))

                if (r_extent is None and not is_allowed_unknown(a1)) or (c_extent is None and not is_allowed_unknown(a2)):
                    return Shape.unknown()

                # Resolve ':' meaning "all rows/cols"
                if isinstance(a1, Colon):
                    r_extent = m
                if isinstance(a2, Colon):
                    c_extent = n

                # Range with unknown extent stays None (already)
                if r_extent == 1 and c_extent == 1:
                    return Shape.scalar()

                return Shape.matrix(r_extent, c_extent)

            if len(args) > 2:
                warnings.append(diag.warn_too_many_indices(line, expr))
                return Shape.unknown()

        return Shape.unknown()

    if isinstance(expr, BinOp):
        op = expr.op
        line = expr.line
        left_shape = eval_expr_ir(expr.left, env, warnings)
        right_shape = eval_expr_ir(expr.right, env, warnings)
        return eval_binop_ir(op, left_shape, right_shape, warnings, expr.left, expr.right, line)

    return Shape.unknown()

def index_arg_to_extent_ir(
    arg: IndexArg,
    env: Env,
    warnings: List[str],
    line: int
) -> Dim:
    """
    Return how many rows/cols this index selects:
      Colon -> unknown extent (resolved later to m/n)
      IndexExpr -> 1 if scalar-shaped else warn + None
      Range -> extent if computable else None
    """
    if isinstance(arg, Colon):
        return None

    if isinstance(arg, Range):
        start_expr = arg.start
        end_expr = arg.end

        start_shape = eval_expr_ir(start_expr, env, warnings)
        end_shape = eval_expr_ir(end_expr, env, warnings)

        if start_shape.is_matrix() or end_shape.is_matrix():
            warnings.append(diag.warn_range_endpoints_must_be_scalar(line, arg, start_shape, end_shape))
            return None

        a = expr_to_dim_ir(start_expr, env)
        b = expr_to_dim_ir(end_expr, env)

        if isinstance(a, int) and isinstance(b, int):
            if b < a:
                warnings.append(diag.warn_invalid_range_end_lt_start(line, arg))
                return None
            return (b - a) + 1

        return None

    if isinstance(arg, IndexExpr):
        s = eval_expr_ir(arg.expr, env, warnings)
        if s.is_matrix():
            warnings.append(diag.warn_non_scalar_index_arg(line, arg, s))
            return None
        return 1

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
    if isinstance(expr, Var):
        return expr.name
    return None


def eval_binop_ir(
    op: str,
    left: Shape,
    right: Shape,
    warnings: List[str],
    left_expr: Expr,
    right_expr: Expr,
    line: int
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

    if left.is_scalar() and not right.is_scalar():
        return right
    if right.is_scalar() and not left.is_scalar():
        return left

    if op in {"+", "-", ".*", "./", "/"}:
        if left.is_unknown() or right.is_unknown():
            return Shape.unknown()

        if left.is_scalar() and right.is_scalar():
            return Shape.scalar()

        if left.is_matrix() and right.is_matrix():
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