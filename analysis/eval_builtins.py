# Ethan Doughty
# eval_builtins.py
"""Builtin function shape inference via dispatch table."""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from ir import Apply
from analysis.context import AnalysisContext
from analysis.dim_extract import expr_to_dim_ir, unwrap_arg
from analysis.intervals import interval_definitely_negative
from runtime.env import Env
from runtime.shapes import Shape, shape_of_zeros, shape_of_ones, add_dim, mul_dim, sub_dim, join_dim, dims_definitely_conflict

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


def _handle_zeros_ones(fname, expr, env, warnings, ctx):
    """zeros(n), zeros(m,n), ones(n), ones(m,n)."""
    # Import here to avoid circular dependency
    from analysis.eval_expr import _get_expr_interval
    import analysis.diagnostics as diag

    if len(expr.args) == 1:
        arg = unwrap_arg(expr.args[0])
        # Check for negative dimension
        dim_interval = _get_expr_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env)
            return shape_of_zeros(d, d) if fname == "zeros" else shape_of_ones(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        arg0 = unwrap_arg(expr.args[0])
        arg1 = unwrap_arg(expr.args[1])
        # Check for negative dimensions
        dim0_interval = _get_expr_interval(arg0, env, ctx)
        dim1_interval = _get_expr_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r_dim = expr_to_dim_ir(arg0, env)
            c_dim = expr_to_dim_ir(arg1, env)
            return shape_of_zeros(r_dim, c_dim) if fname == "zeros" else shape_of_ones(r_dim, c_dim)
        except ValueError:
            pass
    return None


def _handle_matrix_constructor(fname, expr, env, warnings, ctx):
    """eye, rand, randn: eye(n), eye(m,n), etc."""
    # Import here to avoid circular dependency
    from analysis.eval_expr import _get_expr_interval
    import analysis.diagnostics as diag

    if len(expr.args) <= 2:
        if len(expr.args) == 0:
            return Shape.scalar()
        elif len(expr.args) == 1:
            arg = unwrap_arg(expr.args[0])
            # Check for negative dimension
            dim_interval = _get_expr_interval(arg, env, ctx)
            if interval_definitely_negative(dim_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
            try:
                d = expr_to_dim_ir(arg, env)
                return Shape.matrix(d, d)
            except ValueError:
                pass
        elif len(expr.args) == 2:
            arg0 = unwrap_arg(expr.args[0])
            arg1 = unwrap_arg(expr.args[1])
            # Check for negative dimensions
            dim0_interval = _get_expr_interval(arg0, env, ctx)
            dim1_interval = _get_expr_interval(arg1, env, ctx)
            if interval_definitely_negative(dim0_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
            if interval_definitely_negative(dim1_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
            try:
                r = expr_to_dim_ir(arg0, env)
                c = expr_to_dim_ir(arg1, env)
                return Shape.matrix(r, c)
            except ValueError:
                pass
    return None


def _handle_size(fname, expr, env, warnings, ctx):
    """size(x) -> 1x2, size(x, dim) -> scalar."""
    from analysis.eval_expr import eval_expr_ir
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
    from analysis.eval_expr import eval_expr_ir
    if len(expr.args) == 1:
        try:
            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.scalar()
        except ValueError:
            pass
    return None


def _handle_cell_constructor(fname, expr, env, warnings, ctx):
    """cell(n) -> cell[n x n], cell(m,n) -> cell[m x n]."""
    # Import here to avoid circular dependency
    from analysis.eval_expr import _get_expr_interval
    import analysis.diagnostics as diag

    if len(expr.args) == 1:
        arg = unwrap_arg(expr.args[0])
        # Check for negative dimension
        dim_interval = _get_expr_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env)
            return Shape.cell(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        arg0 = unwrap_arg(expr.args[0])
        arg1 = unwrap_arg(expr.args[1])
        # Check for negative dimensions
        dim0_interval = _get_expr_interval(arg0, env, ctx)
        dim1_interval = _get_expr_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r_dim = expr_to_dim_ir(arg0, env)
            c_dim = expr_to_dim_ir(arg1, env)
            return Shape.cell(r_dim, c_dim)
        except ValueError:
            pass
    return None


def _handle_passthrough(fname, expr, env, warnings, ctx):
    """abs(x), sqrt(x) -> same shape as x."""
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        return _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
    return None


def _handle_transpose_fn(fname, expr, env, warnings, ctx):
    """transpose(x) -> swap rows/cols."""
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        if arg_shape.is_matrix():
            return Shape.matrix(arg_shape.cols, arg_shape.rows)
        return arg_shape
    return None


def _handle_scalar_query(fname, expr, env, warnings, ctx):
    """length(x), numel(x), det(x), norm(x) -> scalar."""
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        _ = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        return Shape.scalar()
    return None


def _handle_reshape(fname, expr, env, warnings, ctx):
    """reshape(x, m, n) -> matrix[m x n] with conformability check."""
    from analysis.eval_expr import eval_expr_ir
    from analysis.diagnostics import warn_reshape_mismatch
    if len(expr.args) == 3:
        try:
            input_shape = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
            m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)

            # Conformability check: input element count must equal output element count
            if not input_shape.is_unknown():
                if input_shape.is_scalar():
                    input_count = 1
                elif input_shape.is_matrix():
                    input_count = mul_dim(input_shape.rows, input_shape.cols)
                else:
                    input_count = None  # string/struct/cell/etc -- skip check

                output_count = mul_dim(m, n)

                if input_count is not None and output_count is not None:
                    if dims_definitely_conflict(input_count, output_count):
                        warnings.append(warn_reshape_mismatch(expr.line, input_shape, m, n))

            return Shape.matrix(m, n)
        except ValueError:
            pass
    return None


def _handle_repmat(fname, expr, env, warnings, ctx):
    """repmat(A, m, n) -> matrix[A_rows*m x A_cols*n]."""
    from analysis.eval_expr import eval_expr_ir
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
    from analysis.eval_expr import _eval_index_arg_to_shape
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
    from analysis.eval_expr import _eval_index_arg_to_shape
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
    from analysis.eval_expr import eval_expr_ir
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


def _handle_reduction(fname, expr, env, warnings, ctx):
    """sum, prod, mean, any, all: reduce along dimension (default dim 1).

    1-arg: reduce along dim 1. scalar->scalar, matrix[m x n]->matrix[1 x n].
    2-arg: reduce along specified dim (1 or 2). Non-concrete dim->unknown.
    3+ args: return None (falls through to unknown).
    """
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        if arg_shape.is_scalar():
            return Shape.scalar()
        if arg_shape.is_matrix():
            return Shape.matrix(1, arg_shape.cols)
        return Shape.unknown()
    elif len(expr.args) == 2:
        arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        try:
            dim_val = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
            # Only handle concrete int dims 1 or 2
            if dim_val == 1:
                if arg_shape.is_matrix():
                    return Shape.matrix(1, arg_shape.cols)
                return Shape.unknown()
            elif dim_val == 2:
                if arg_shape.is_matrix():
                    return Shape.matrix(arg_shape.rows, 1)
                return Shape.unknown()
            else:
                return None  # Non-1/2 dim -> unknown
        except ValueError:
            pass
    return None


def _handle_minmax(fname, expr, env, warnings, ctx):
    """min, max: dispatch by arg count.

    1-arg: delegate to _handle_reduction.
    2-arg: delegate to _handle_elementwise_2arg.
    3+ args: return None (falls through to unknown).
    """
    if len(expr.args) == 1:
        return _handle_reduction(fname, expr, env, warnings, ctx)
    elif len(expr.args) == 2:
        return _handle_elementwise_2arg(fname, expr, env, warnings, ctx)
    return None


def _handle_elementwise_2arg(fname, expr, env, warnings, ctx):
    """mod, rem, atan2: elementwise binary operation.

    Requires exactly 2 args.
    Scalar broadcasting: scalar op matrix -> matrix.
    Both matrix: join dims if compatible, unknown if dims definitely conflict.
    """
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) != 2:
        return None

    s1 = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
    s2 = _eval_index_arg_to_shape(expr.args[1], env, warnings, ctx)

    # Scalar broadcasting
    if s1.is_scalar():
        return s2
    if s2.is_scalar():
        return s1

    # Both matrix: check compatibility
    if s1.is_matrix() and s2.is_matrix():
        r1, c1 = s1.rows, s1.cols
        r2, c2 = s2.rows, s2.cols

        # Check for definite conflicts
        if dims_definitely_conflict(r1, r2) or dims_definitely_conflict(c1, c2):
            return Shape.unknown()

        # Join dimensions
        return Shape.matrix(join_dim(r1, r2), join_dim(c1, c2))

    return Shape.unknown()


def _handle_diff(fname, expr, env, warnings, ctx):
    """diff: differentiation along dimension.

    1-arg only (multi-arg deferred, returns None).
    scalar -> scalar (approximation; MATLAB returns []).
    matrix[1 x n] -> matrix[1 x (n-1)] (row vector: diff along dim 2).
    matrix[m x n] -> matrix[(m-1) x n] (general: diff along dim 1).
    """
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) != 1:
        return None

    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)

    if arg_shape.is_scalar():
        return Shape.scalar()

    if arg_shape.is_matrix():
        r, c = arg_shape.rows, arg_shape.cols
        # Row vector: diff along dim 2
        if r == 1:
            return Shape.matrix(1, sub_dim(c, 1))
        # General matrix: diff along dim 1
        return Shape.matrix(sub_dim(r, 1), c)

    return Shape.unknown()


def _handle_kron(fname, expr, env, warnings, ctx):
    """kron(A, B) -> matrix[(m*p) x (n*q)] where A is m×n, B is p×q."""
    from analysis.eval_expr import eval_expr_ir
    if len(expr.args) != 2:
        return None
    try:
        s1 = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
        s2 = eval_expr_ir(unwrap_arg(expr.args[1]), env, warnings, ctx)
    except ValueError:
        return None

    if s1.is_unknown() or s2.is_unknown():
        return Shape.unknown()

    r1, c1 = (1, 1) if s1.is_scalar() else (s1.rows, s1.cols) if s1.is_matrix() else (None, None)
    r2, c2 = (1, 1) if s2.is_scalar() else (s2.rows, s2.cols) if s2.is_matrix() else (None, None)

    if r1 is None or r2 is None:
        return Shape.unknown()

    return Shape.matrix(mul_dim(r1, r2), mul_dim(c1, c2))


def _handle_blkdiag(fname, expr, env, warnings, ctx):
    """blkdiag(A, B, ...) -> matrix[sum(rows) x sum(cols)]."""
    from analysis.eval_expr import eval_expr_ir
    if len(expr.args) == 0:
        return None
    total_rows = 0
    total_cols = 0
    for arg in expr.args:
        try:
            s = eval_expr_ir(unwrap_arg(arg), env, warnings, ctx)
        except ValueError:
            return None
        if s.is_unknown():
            return Shape.unknown()
        if s.is_scalar():
            r, c = 1, 1
        elif s.is_matrix():
            r, c = s.rows, s.cols
        else:
            return Shape.unknown()
        total_rows = add_dim(total_rows, r)
        total_cols = add_dim(total_cols, c)
    return Shape.matrix(total_rows, total_cols)


def _handle_string_return(fname, expr, env, warnings, ctx):
    """String conversion builtins: num2str, int2str, mat2str, char, string, sprintf.

    Evaluate all args for warning propagation, return Shape.string().
    """
    from analysis.eval_expr import _eval_index_arg_to_shape
    # Evaluate all args for side effects (warning propagation)
    for arg in expr.args:
        _ = _eval_index_arg_to_shape(arg, env, warnings, ctx)
    return Shape.string()


def _handle_type_cast(fname, expr, env, warnings, ctx):
    """Type cast builtins: double, single, int8, int16, int32, int64, uint8, uint16, uint32, uint64, logical, complex.

    Delegate to _handle_passthrough. Separate function for semantic clarity (future type tracking).
    """
    return _handle_passthrough(fname, expr, env, warnings, ctx)


def _handle_find(fname, expr, env, warnings, ctx):
    """find(x) -> matrix[1 x None] (row vector, unknown length)."""
    from analysis.eval_expr import _eval_index_arg_to_shape
    if len(expr.args) == 1:
        _ = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
        return Shape.matrix(1, None)
    return None


def _handle_cat(fname, expr, env, warnings, ctx):
    """cat(dim, A, B, ...) -> concatenation along specified dimension.

    cat(1, ...): rows add via add_dim, cols join via join_dim.
    cat(2, ...): rows join, cols add.
    Non-concrete dim or dim not in {1,2}: return None (falls through to unknown).
    """
    from analysis.eval_expr import eval_expr_ir
    if len(expr.args) < 2:
        return None

    # First arg is the dimension
    try:
        dim_val = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
    except ValueError:
        return None

    # Only handle concrete dims 1 or 2
    if dim_val not in [1, 2]:
        return None

    # Evaluate all matrix args
    shapes = []
    for arg in expr.args[1:]:
        try:
            s = eval_expr_ir(unwrap_arg(arg), env, warnings, ctx)
        except ValueError:
            return None
        if s.is_unknown():
            return Shape.unknown()
        shapes.append(s)

    if not shapes:
        return None

    # Convert scalars to 1x1 matrices for uniform handling
    def normalize_shape(s):
        if s.is_scalar():
            return (1, 1)
        elif s.is_matrix():
            return (s.rows, s.cols)
        else:
            return None

    normalized = [normalize_shape(s) for s in shapes]
    if any(n is None for n in normalized):
        return Shape.unknown()

    # Fold across all args
    result_rows, result_cols = normalized[0]
    for r, c in normalized[1:]:
        if dim_val == 1:
            # cat(1, ...): rows add, cols join
            result_rows = add_dim(result_rows, r)
            result_cols = join_dim(result_cols, c)
        else:  # dim_val == 2
            # cat(2, ...): rows join, cols add
            result_rows = join_dim(result_rows, r)
            result_cols = add_dim(result_cols, c)

    return Shape.matrix(result_rows, result_cols)


def _handle_randi(fname, expr, env, warnings, ctx):
    """randi(imax), randi(imax, n), randi(imax, m, n).

    Like _handle_matrix_constructor but first arg is imax (value range), not a dimension.
    - randi(imax) -> scalar
    - randi(imax, n) -> matrix[n x n]
    - randi(imax, m, n) -> matrix[m x n]
    """
    from analysis.eval_expr import eval_expr_ir, _get_expr_interval
    import analysis.diagnostics as diag

    if len(expr.args) < 1:
        return None

    # Evaluate first arg (imax) for warning propagation, but don't use it for shape
    try:
        _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return None

    # Remaining args are dimensions, like _handle_matrix_constructor
    if len(expr.args) == 1:
        return Shape.scalar()
    elif len(expr.args) == 2:
        arg = unwrap_arg(expr.args[1])
        # Check for negative dimension
        dim_interval = _get_expr_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env)
            return Shape.matrix(d, d)
        except ValueError:
            pass
    elif len(expr.args) == 3:
        arg0 = unwrap_arg(expr.args[1])
        arg1 = unwrap_arg(expr.args[2])
        # Check for negative dimensions
        dim0_interval = _get_expr_interval(arg0, env, ctx)
        dim1_interval = _get_expr_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r = expr_to_dim_ir(arg0, env)
            c = expr_to_dim_ir(arg1, env)
            return Shape.matrix(r, c)
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
    'true': _handle_matrix_constructor,
    'false': _handle_matrix_constructor,
    'nan': _handle_matrix_constructor,
    'inf': _handle_matrix_constructor,
    'size': _handle_size,
    'isscalar': _handle_scalar_predicate,
    'iscell': _handle_scalar_predicate,
    'isempty': _handle_scalar_predicate,
    'isnumeric': _handle_scalar_predicate,
    'islogical': _handle_scalar_predicate,
    'ischar': _handle_scalar_predicate,
    'isnan': _handle_scalar_predicate,
    'isinf': _handle_scalar_predicate,
    'isfinite': _handle_scalar_predicate,
    'issymmetric': _handle_scalar_predicate,
    'isstruct': _handle_scalar_predicate,
    'isreal': _handle_scalar_predicate,
    'issparse': _handle_scalar_predicate,
    'isvector': _handle_scalar_predicate,
    'isinteger': _handle_scalar_predicate,
    'isfloat': _handle_scalar_predicate,
    'isstring': _handle_scalar_predicate,
    'issorted': _handle_scalar_predicate,
    'cell': _handle_cell_constructor,
    'abs': _handle_passthrough,
    'sqrt': _handle_passthrough,
    'sin': _handle_passthrough,
    'cos': _handle_passthrough,
    'tan': _handle_passthrough,
    'asin': _handle_passthrough,
    'acos': _handle_passthrough,
    'atan': _handle_passthrough,
    'tanh': _handle_passthrough,
    'cosh': _handle_passthrough,
    'sinh': _handle_passthrough,
    'atanh': _handle_passthrough,
    'acosh': _handle_passthrough,
    'asinh': _handle_passthrough,
    'conj': _handle_passthrough,
    'not': _handle_passthrough,
    'flipud': _handle_passthrough,
    'fliplr': _handle_passthrough,
    'triu': _handle_passthrough,
    'tril': _handle_passthrough,
    'sort': _handle_passthrough,
    'unique': _handle_passthrough,
    'exp': _handle_passthrough,
    'log': _handle_passthrough,
    'log2': _handle_passthrough,
    'log10': _handle_passthrough,
    'ceil': _handle_passthrough,
    'floor': _handle_passthrough,
    'round': _handle_passthrough,
    'sign': _handle_passthrough,
    'real': _handle_passthrough,
    'imag': _handle_passthrough,
    'cumsum': _handle_passthrough,
    'cumprod': _handle_passthrough,
    'transpose': _handle_transpose_fn,
    'length': _handle_scalar_query,
    'numel': _handle_scalar_query,
    'det': _handle_scalar_query,
    'norm': _handle_scalar_query,
    'trace': _handle_scalar_query,
    'rank': _handle_scalar_query,
    'cond': _handle_scalar_query,
    'rcond': _handle_scalar_query,
    'nnz': _handle_scalar_query,
    'sprank': _handle_scalar_query,
    'reshape': _handle_reshape,
    'repmat': _handle_repmat,
    'diag': _handle_diag,
    'inv': _handle_inv,
    'linspace': _handle_linspace,
    'sum': _handle_reduction,
    'prod': _handle_reduction,
    'mean': _handle_reduction,
    'any': _handle_reduction,
    'all': _handle_reduction,
    'median': _handle_reduction,
    'var': _handle_reduction,
    'std': _handle_reduction,
    'min': _handle_minmax,
    'max': _handle_minmax,
    'mod': _handle_elementwise_2arg,
    'rem': _handle_elementwise_2arg,
    'atan2': _handle_elementwise_2arg,
    'power': _handle_elementwise_2arg,
    'hypot': _handle_elementwise_2arg,
    'xor': _handle_elementwise_2arg,
    'diff': _handle_diff,
    'kron': _handle_kron,
    'blkdiag': _handle_blkdiag,
    'double': _handle_type_cast,
    'single': _handle_type_cast,
    'int8': _handle_type_cast,
    'int16': _handle_type_cast,
    'int32': _handle_type_cast,
    'int64': _handle_type_cast,
    'uint8': _handle_type_cast,
    'uint16': _handle_type_cast,
    'uint32': _handle_type_cast,
    'uint64': _handle_type_cast,
    'logical': _handle_type_cast,
    'complex': _handle_type_cast,
    'num2str': _handle_string_return,
    'int2str': _handle_string_return,
    'mat2str': _handle_string_return,
    'char': _handle_string_return,
    'string': _handle_string_return,
    'sprintf': _handle_string_return,
    'randi': _handle_randi,
    'find': _handle_find,
    'cat': _handle_cat,
}


def eval_builtin_call(fname: str, expr: Apply, env: Env, warnings: List['Diagnostic'], ctx: AnalysisContext) -> Shape:
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
