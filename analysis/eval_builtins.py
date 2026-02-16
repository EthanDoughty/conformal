# Ethan Doughty
# eval_builtins.py
"""Builtin function shape inference via dispatch table."""

from __future__ import annotations
from typing import List, Optional

from ir import Apply, IndexArg
from analysis.context import AnalysisContext
from analysis.dim_extract import expr_to_dim_ir, unwrap_arg
from runtime.env import Env
from runtime.shapes import Shape, shape_of_zeros, shape_of_ones, mul_dim


def _handle_zeros_ones(fname, expr, env, warnings, ctx):
    """zeros(n), zeros(m,n), ones(n), ones(m,n)."""
    if len(expr.args) == 1:
        try:
            d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
            return shape_of_zeros(d, d) if fname == "zeros" else shape_of_ones(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        try:
            r_dim = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
            c_dim = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            return shape_of_zeros(r_dim, c_dim) if fname == "zeros" else shape_of_ones(r_dim, c_dim)
        except ValueError:
            pass
    return None


def _handle_matrix_constructor(fname, expr, env, warnings, ctx):
    """eye, rand, randn: eye(n), eye(m,n), etc."""
    if len(expr.args) <= 2:
        if len(expr.args) == 0:
            return Shape.scalar()
        elif len(expr.args) == 1:
            try:
                d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                return Shape.matrix(d, d)
            except ValueError:
                pass
        elif len(expr.args) == 2:
            try:
                r = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                c = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
                return Shape.matrix(r, c)
            except ValueError:
                pass
    return None


def _handle_size(fname, expr, env, warnings, ctx):
    """size(x) -> 1x2, size(x, dim) -> scalar."""
    from analysis.analysis_ir import eval_expr_ir
    if len(expr.args) == 1:
        try:
            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.matrix(1, 2)
        except ValueError:
            pass
    elif len(expr.args) == 2:
        try:
            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.scalar()
        except ValueError:
            pass
    return None


def _handle_scalar_predicate(fname, expr, env, warnings, ctx):
    """isscalar(x), iscell(x) -> scalar."""
    from analysis.analysis_ir import eval_expr_ir
    if len(expr.args) == 1:
        try:
            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.scalar()
        except ValueError:
            pass
    return None


def _handle_cell_constructor(fname, expr, env, warnings, ctx):
    """cell(n) -> cell[n x n], cell(m,n) -> cell[m x n]."""
    if len(expr.args) == 1:
        try:
            d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
            return Shape.cell(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        try:
            r_dim = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
            c_dim = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            return Shape.cell(r_dim, c_dim)
        except ValueError:
            pass
    return None


def _handle_passthrough(fname, expr, env, warnings, ctx):
    """abs(x), sqrt(x) -> same shape as x."""
    from analysis.analysis_ir import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        return _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
    return None


def _handle_transpose_fn(fname, expr, env, warnings, ctx):
    """transpose(x) -> swap rows/cols."""
    from analysis.analysis_ir import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        if arg_shape.is_matrix():
            return Shape.matrix(arg_shape.cols, arg_shape.rows)
        return arg_shape
    return None


def _handle_scalar_query(fname, expr, env, warnings, ctx):
    """length(x), numel(x), det(x), norm(x) -> scalar."""
    from analysis.analysis_ir import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        _ = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        return Shape.scalar()
    return None


def _handle_reshape(fname, expr, env, warnings, ctx):
    """reshape(x, m, n) -> matrix[m x n]."""
    from analysis.analysis_ir import eval_expr_ir
    if len(expr.args) == 3:
        try:
            _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)
            return Shape.matrix(m, n)
        except ValueError:
            pass
    return None


def _handle_repmat(fname, expr, env, warnings, ctx):
    """repmat(A, m, n) -> matrix[A_rows*m x A_cols*n]."""
    from analysis.analysis_ir import eval_expr_ir
    if len(expr.args) == 3:
        try:
            a_shape = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)
            if a_shape.is_unknown():
                return Shape.unknown()
            if a_shape.is_scalar():
                a_rows, a_cols = 1, 1
            else:
                a_rows, a_cols = a_shape.rows, a_shape.cols
            return Shape.matrix(mul_dim(a_rows, m), mul_dim(a_cols, n))
        except ValueError:
            pass
    return None


def _handle_diag(fname, expr, env, warnings, ctx):
    """diag(x): vector->matrix, matrix->vector."""
    from analysis.analysis_ir import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        if arg_shape.is_scalar():
            return Shape.matrix(1, 1)
        if arg_shape.is_matrix():
            r, c = arg_shape.rows, arg_shape.cols
            if r == 1:
                return Shape.matrix(c, c)
            if c == 1:
                return Shape.matrix(r, r)
            return Shape.matrix(None, 1)
        return Shape.unknown()
    return None


def _handle_inv(fname, expr, env, warnings, ctx):
    """inv(x): pass-through for square matrices."""
    from analysis.analysis_ir import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        if arg_shape.is_matrix():
            r, c = arg_shape.rows, arg_shape.cols
            if r == c:
                return Shape.matrix(r, c)
            return Shape.unknown()
        return Shape.unknown()
    return None


def _handle_linspace(fname, expr, env, warnings, ctx):
    """linspace(a,b) -> 1x100, linspace(a,b,n) -> 1xn."""
    from analysis.analysis_ir import eval_expr_ir
    if len(expr.args) == 2:
        try:
            _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            _ = eval_expr_ir(unwrap_arg(expr.args[1]), env, warnings, ctx)
            return Shape.matrix(1, 100)
        except ValueError:
            pass
    elif len(expr.args) == 3:
        try:
            _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            _ = eval_expr_ir(unwrap_arg(expr.args[1]), env, warnings, ctx)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)
            return Shape.matrix(1, n)
        except ValueError:
            pass
    return None


# Dispatch table: builtin name -> handler function
BUILTIN_HANDLERS = {
    'zeros': _handle_zeros_ones,
    'ones': _handle_zeros_ones,
    'eye': _handle_matrix_constructor,
    'rand': _handle_matrix_constructor,
    'randn': _handle_matrix_constructor,
    'size': _handle_size,
    'isscalar': _handle_scalar_predicate,
    'iscell': _handle_scalar_predicate,
    'cell': _handle_cell_constructor,
    'abs': _handle_passthrough,
    'sqrt': _handle_passthrough,
    'transpose': _handle_transpose_fn,
    'length': _handle_scalar_query,
    'numel': _handle_scalar_query,
    'det': _handle_scalar_query,
    'norm': _handle_scalar_query,
    'reshape': _handle_reshape,
    'repmat': _handle_repmat,
    'diag': _handle_diag,
    'inv': _handle_inv,
    'linspace': _handle_linspace,
}


def eval_builtin_call(fname: str, expr: Apply, env: Env, warnings: List[str], ctx: AnalysisContext) -> Shape:
    """Dispatch builtin function call to appropriate handler.

    Args:
        fname: Builtin function name
        expr: Apply expression node
        env: Current environment
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        Inferred shape of the result
    """
    handler = BUILTIN_HANDLERS.get(fname)
    if handler:
        result = handler(fname, expr, env, warnings, ctx)
        if result is not None:
            return result
    # Known builtin without a matching shape rule: return unknown silently
    return Shape.unknown()
