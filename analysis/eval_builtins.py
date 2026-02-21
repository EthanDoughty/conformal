# Ethan Doughty
# eval_builtins.py
"""Builtin function shape inference via dispatch table."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

from ir import Apply
from analysis.context import AnalysisContext
from analysis.dim_extract import expr_to_dim_ir, unwrap_arg
from analysis.intervals import interval_definitely_negative
from runtime.env import Env
from runtime.shapes import Shape, shape_of_zeros, shape_of_ones, add_dim, mul_dim, sub_dim, join_dim, dims_definitely_conflict

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


@dataclass
class BuiltinEvalContext:
    """Callbacks injected into builtin handlers to break the circular import.

    eval_builtins -> eval_expr -> eval_builtins is avoided by passing these
    three functions as data rather than importing them at module level.
    """
    eval_expr: Callable    # eval_expr_ir(expr, env, warnings, ctx, container_shape=None) -> Shape
    eval_arg: Callable     # _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=None) -> Shape
    get_interval: Callable  # _get_expr_interval(expr, env, ctx) -> Optional[Interval]


def _handle_zeros_ones(fname, expr, env, warnings, ctx, evals):
    """zeros(n), zeros(m,n), ones(n), ones(m,n)."""
    import analysis.diagnostics as diag

    if len(expr.args) == 1:
        arg = unwrap_arg(expr.args[0])
        # Check for negative dimension
        dim_interval = evals.get_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env, ctx)
            return shape_of_zeros(d, d) if fname == "zeros" else shape_of_ones(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        arg0 = unwrap_arg(expr.args[0])
        arg1 = unwrap_arg(expr.args[1])
        # Check for negative dimensions
        dim0_interval = evals.get_interval(arg0, env, ctx)
        dim1_interval = evals.get_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r_dim = expr_to_dim_ir(arg0, env, ctx)
            c_dim = expr_to_dim_ir(arg1, env, ctx)
            return shape_of_zeros(r_dim, c_dim) if fname == "zeros" else shape_of_ones(r_dim, c_dim)
        except ValueError:
            pass
    return None


def _handle_matrix_constructor(fname, expr, env, warnings, ctx, evals):
    """eye, rand, randn: eye(n), eye(m,n), etc."""
    import analysis.diagnostics as diag

    if len(expr.args) <= 2:
        if len(expr.args) == 0:
            return Shape.scalar()
        elif len(expr.args) == 1:
            arg = unwrap_arg(expr.args[0])
            # Check for negative dimension
            dim_interval = evals.get_interval(arg, env, ctx)
            if interval_definitely_negative(dim_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
            try:
                d = expr_to_dim_ir(arg, env, ctx)
                return Shape.matrix(d, d)
            except ValueError:
                pass
        elif len(expr.args) == 2:
            arg0 = unwrap_arg(expr.args[0])
            arg1 = unwrap_arg(expr.args[1])
            # Check for negative dimensions
            dim0_interval = evals.get_interval(arg0, env, ctx)
            dim1_interval = evals.get_interval(arg1, env, ctx)
            if interval_definitely_negative(dim0_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
            if interval_definitely_negative(dim1_interval):
                warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
            try:
                r = expr_to_dim_ir(arg0, env, ctx)
                c = expr_to_dim_ir(arg1, env, ctx)
                return Shape.matrix(r, c)
            except ValueError:
                pass
    return None


def _handle_size(fname, expr, env, warnings, ctx, evals):
    """size(x) -> 1x2, size(x, dim) -> scalar."""
    if len(expr.args) == 1:
        try:
            evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.matrix(1, 2)
        except ValueError:
            pass
    elif len(expr.args) == 2:
        try:
            evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.scalar()
        except ValueError:
            pass
    return None


def _handle_scalar_predicate(fname, expr, env, warnings, ctx, evals):
    """isscalar(x), iscell(x) -> scalar."""
    if len(expr.args) == 1:
        try:
            evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            return Shape.scalar()
        except ValueError:
            pass
    return None


def _handle_cell_constructor(fname, expr, env, warnings, ctx, evals):
    """cell(n) -> cell[n x n], cell(m,n) -> cell[m x n]."""
    import analysis.diagnostics as diag

    if len(expr.args) == 1:
        arg = unwrap_arg(expr.args[0])
        # Check for negative dimension
        dim_interval = evals.get_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env, ctx)
            return Shape.cell(d, d)
        except ValueError:
            pass
    if len(expr.args) == 2:
        arg0 = unwrap_arg(expr.args[0])
        arg1 = unwrap_arg(expr.args[1])
        # Check for negative dimensions
        dim0_interval = evals.get_interval(arg0, env, ctx)
        dim1_interval = evals.get_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r_dim = expr_to_dim_ir(arg0, env, ctx)
            c_dim = expr_to_dim_ir(arg1, env, ctx)
            return Shape.cell(r_dim, c_dim)
        except ValueError:
            pass
    return None


PASSTHROUGH_BUILTINS = frozenset({
    'abs', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'tanh', 'cosh', 'sinh', 'atanh', 'acosh', 'asinh', 'conj', 'not',
    'flipud', 'fliplr', 'triu', 'tril', 'sort', 'unique',
    'exp', 'log', 'log2', 'log10', 'ceil', 'floor', 'round', 'sign',
    'real', 'imag', 'cumsum', 'cumprod',
    'expm', 'logm', 'sqrtm', 'circshift', 'null', 'orth',
    'sgolayfilt', 'squeeze', 'fftshift', 'ifftshift', 'unwrap',
    'deg2rad', 'rad2deg', 'angle',
})

SCALAR_PREDICATE_BUILTINS = frozenset({
    'isscalar', 'iscell', 'isempty', 'isnumeric', 'islogical', 'ischar',
    'isnan', 'isinf', 'isfinite', 'issymmetric', 'isstruct', 'isreal',
    'issparse', 'isvector', 'isinteger', 'isfloat', 'isstring', 'issorted',
    'isfield',
})

TYPE_CAST_BUILTINS = frozenset({
    'double', 'single', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64', 'logical', 'complex', 'typecast',
})

REDUCTION_BUILTINS = frozenset({
    'sum', 'prod', 'mean', 'any', 'all', 'median', 'var', 'std',
    'nanmean', 'nansum', 'nanstd', 'nanmin', 'nanmax',
})

SCALAR_QUERY_BUILTINS = frozenset({
    'length', 'numel', 'det', 'norm', 'trace', 'rank', 'cond', 'rcond',
    'nnz', 'sprank', 'str2double',
})

MATRIX_CONSTRUCTOR_BUILTINS = frozenset({
    'eye', 'rand', 'randn', 'true', 'false', 'nan', 'NaN', 'inf', 'Inf',
})

STRING_RETURN_BUILTINS = frozenset({
    'num2str', 'int2str', 'mat2str', 'char', 'string', 'sprintf', 'fullfile',
})

SCALAR_NARY_BUILTINS = frozenset({
    'strcmpi', 'strcmp', 'exist',
})

# Sanity check: no builtin should appear in more than one declarative set
_all_declarative = (
    PASSTHROUGH_BUILTINS | SCALAR_PREDICATE_BUILTINS | TYPE_CAST_BUILTINS |
    REDUCTION_BUILTINS | SCALAR_QUERY_BUILTINS | MATRIX_CONSTRUCTOR_BUILTINS |
    STRING_RETURN_BUILTINS | SCALAR_NARY_BUILTINS
)
assert (
    len(_all_declarative) ==
    len(PASSTHROUGH_BUILTINS) + len(SCALAR_PREDICATE_BUILTINS) +
    len(TYPE_CAST_BUILTINS) + len(REDUCTION_BUILTINS) +
    len(SCALAR_QUERY_BUILTINS) + len(MATRIX_CONSTRUCTOR_BUILTINS) +
    len(STRING_RETURN_BUILTINS) + len(SCALAR_NARY_BUILTINS)
), "Declarative builtin sets have overlapping entries"


def _handle_passthrough(fname, expr, env, warnings, ctx, evals):
    """abs(x), sqrt(x) -> same shape as x."""
    if len(expr.args) == 1:
        return evals.eval_arg(expr.args[0], env, warnings, ctx)
    return None


def _handle_transpose_fn(fname, expr, env, warnings, ctx, evals):
    """transpose(x) -> swap rows/cols."""
    if len(expr.args) == 1:
        arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
        if arg_shape.is_matrix():
            return Shape.matrix(arg_shape.cols, arg_shape.rows)
        return arg_shape
    return None


def _handle_scalar_query(fname, expr, env, warnings, ctx, evals):
    """length(x), numel(x), det(x), norm(x) -> scalar."""
    if len(expr.args) == 1:
        _ = evals.eval_arg(expr.args[0], env, warnings, ctx)
        return Shape.scalar()
    return None


def _handle_scalar_nary(fname, expr, env, warnings, ctx, evals):
    """strcmp, strcmpi, exist, etc. -> always scalar regardless of arg count."""
    return Shape.scalar()


def _handle_reshape(fname, expr, env, warnings, ctx, evals):
    """reshape(x, m, n) -> matrix[m x n] with conformability check."""
    from analysis.diagnostics import warn_reshape_mismatch
    if len(expr.args) == 3:
        try:
            input_shape = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env, ctx)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env, ctx)

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
                        from analysis.witness import ConflictSite
                        ctx.conflict_sites.append(ConflictSite(
                            dim_a=input_count, dim_b=output_count,
                            line=expr.line, warning_code="W_RESHAPE_MISMATCH",
                            constraints_snapshot=frozenset(ctx.constraints),
                            scalar_bindings_snapshot=tuple(sorted(ctx.scalar_bindings.items())),
                            value_ranges_snapshot=tuple(sorted(
                                (k, (v.lo, v.hi)) for k, v in ctx.value_ranges.items()
                            )),
                            path_snapshot=tuple(ctx.path_constraints.snapshot()),
                        ))
                        warnings.append(warn_reshape_mismatch(expr.line, input_shape, m, n))

            return Shape.matrix(m, n)
        except ValueError:
            pass
    return None


def _handle_repmat(fname, expr, env, warnings, ctx, evals):
    """repmat(A, m, n) -> matrix[A_rows*m x A_cols*n]."""
    if len(expr.args) == 3:
        try:
            a_shape = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env, ctx)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env, ctx)
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


def _handle_diag(fname, expr, env, warnings, ctx, evals):
    """diag(x): vector->matrix, matrix->vector."""
    if len(expr.args) == 1:
        arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
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


def _handle_inv(fname, expr, env, warnings, ctx, evals):
    """inv(x): pass-through for square matrices."""
    if len(expr.args) == 1:
        arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
        if arg_shape.is_matrix():
            r, c = arg_shape.rows, arg_shape.cols
            if r == c:
                return Shape.matrix(r, c)
            return Shape.unknown()
        return Shape.unknown()
    return None


def _handle_linspace(fname, expr, env, warnings, ctx, evals):
    """linspace(a,b) -> 1x100, linspace(a,b,n) -> 1xn."""
    if len(expr.args) == 2:
        try:
            _ = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            _ = evals.eval_expr(unwrap_arg(expr.args[1]), env, warnings, ctx)
            return Shape.matrix(1, 100)
        except ValueError:
            pass
    elif len(expr.args) == 3:
        try:
            _ = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            _ = evals.eval_expr(unwrap_arg(expr.args[1]), env, warnings, ctx)
            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env, ctx)
            return Shape.matrix(1, n)
        except ValueError:
            pass
    return None


def _handle_reduction(fname, expr, env, warnings, ctx, evals):
    """sum, prod, mean, any, all: reduce along dimension (default dim 1).

    1-arg: reduce along dim 1. scalar->scalar, matrix[m x n]->matrix[1 x n].
    2-arg: reduce along specified dim (1 or 2). Non-concrete dim->unknown.
    3+ args: return None (falls through to unknown).
    """
    if len(expr.args) == 1:
        arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
        if arg_shape.is_scalar():
            return Shape.scalar()
        if arg_shape.is_matrix():
            return Shape.matrix(1, arg_shape.cols)
        return Shape.unknown()
    elif len(expr.args) == 2:
        arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
        try:
            dim_val = expr_to_dim_ir(unwrap_arg(expr.args[1]), env, ctx)
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


def _handle_minmax(fname, expr, env, warnings, ctx, evals):
    """min, max: dispatch by arg count.

    1-arg: delegate to _handle_reduction.
    2-arg: delegate to _handle_elementwise_2arg.
    3+ args: return None (falls through to unknown).
    """
    if len(expr.args) == 1:
        return _handle_reduction(fname, expr, env, warnings, ctx, evals)
    elif len(expr.args) == 2:
        return _handle_elementwise_2arg(fname, expr, env, warnings, ctx, evals)
    return None


def _handle_elementwise_2arg(fname, expr, env, warnings, ctx, evals):
    """mod, rem, atan2: elementwise binary operation.

    Requires exactly 2 args.
    Scalar broadcasting: scalar op matrix -> matrix.
    Both matrix: join dims if compatible, unknown if dims definitely conflict.
    """
    if len(expr.args) != 2:
        return None

    s1 = evals.eval_arg(expr.args[0], env, warnings, ctx)
    s2 = evals.eval_arg(expr.args[1], env, warnings, ctx)

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


def _handle_diff(fname, expr, env, warnings, ctx, evals):
    """diff: differentiation along dimension.

    1-arg only (multi-arg deferred, returns None).
    scalar -> scalar (approximation; MATLAB returns []).
    matrix[1 x n] -> matrix[1 x (n-1)] (row vector: diff along dim 2).
    matrix[m x n] -> matrix[(m-1) x n] (general: diff along dim 1).
    """
    if len(expr.args) != 1:
        return None

    arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)

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


def _handle_kron(fname, expr, env, warnings, ctx, evals):
    """kron(A, B) -> matrix[(m*p) x (n*q)] where A is m×n, B is p×q."""
    if len(expr.args) != 2:
        return None
    try:
        s1 = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
        s2 = evals.eval_expr(unwrap_arg(expr.args[1]), env, warnings, ctx)
    except ValueError:
        return None

    if s1.is_unknown() or s2.is_unknown():
        return Shape.unknown()

    r1, c1 = (1, 1) if s1.is_scalar() else (s1.rows, s1.cols) if s1.is_matrix() else (None, None)
    r2, c2 = (1, 1) if s2.is_scalar() else (s2.rows, s2.cols) if s2.is_matrix() else (None, None)

    if r1 is None or r2 is None:
        return Shape.unknown()

    return Shape.matrix(mul_dim(r1, r2), mul_dim(c1, c2))


def _handle_blkdiag(fname, expr, env, warnings, ctx, evals):
    """blkdiag(A, B, ...) -> matrix[sum(rows) x sum(cols)]."""
    if len(expr.args) == 0:
        return None
    total_rows = 0
    total_cols = 0
    for arg in expr.args:
        try:
            s = evals.eval_expr(unwrap_arg(arg), env, warnings, ctx)
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


def _handle_string_return(fname, expr, env, warnings, ctx, evals):
    """String conversion builtins: num2str, int2str, mat2str, char, string, sprintf.

    Evaluate all args for warning propagation, return Shape.string().
    """
    # Evaluate all args for side effects (warning propagation)
    for arg in expr.args:
        _ = evals.eval_arg(arg, env, warnings, ctx)
    return Shape.string()


def _handle_type_cast(fname, expr, env, warnings, ctx, evals):
    """Type cast builtins: double, single, int8, int16, int32, int64, uint8, uint16, uint32, uint64, logical, complex.

    Delegate to _handle_passthrough. Separate function for semantic clarity (future type tracking).
    """
    return _handle_passthrough(fname, expr, env, warnings, ctx, evals)


def _handle_find(fname, expr, env, warnings, ctx, evals):
    """find(x) -> matrix[1 x None] (row vector, unknown length)."""
    if len(expr.args) == 1:
        _ = evals.eval_arg(expr.args[0], env, warnings, ctx)
        return Shape.matrix(1, None)
    return None


def _handle_eig_single(fname, expr, env, warnings, ctx, evals):
    """eig(A) -> matrix[n x 1] for n x n input."""
    if len(expr.args) != 1:
        return None
    try:
        s = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return None
    if s.is_scalar():
        return Shape.scalar()
    if s.is_matrix():
        r, c = s.rows, s.cols
        # For square matrix, return eigenvalue vector
        if r == c:
            return Shape.matrix(r, 1)
        # Non-square: return unknown-length vector
        return Shape.matrix(None, 1)
    return Shape.unknown()


def _handle_svd_single(fname, expr, env, warnings, ctx, evals):
    """svd(A) -> matrix[None x 1] (singular value vector)."""
    if len(expr.args) != 1:
        return None
    try:
        _ = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return None
    # Singular values vector length = min(m, n), not expressible
    return Shape.matrix(None, 1)


def _handle_cat(fname, expr, env, warnings, ctx, evals):
    """cat(dim, A, B, ...) -> concatenation along specified dimension.

    cat(1, ...): rows add via add_dim, cols join via join_dim.
    cat(2, ...): rows join, cols add.
    Non-concrete dim or dim not in {1,2}: return None (falls through to unknown).
    """
    if len(expr.args) < 2:
        return None

    # First arg is the dimension
    try:
        dim_val = expr_to_dim_ir(unwrap_arg(expr.args[0]), env, ctx)
    except ValueError:
        return None

    # Only handle concrete dims 1 or 2
    if dim_val not in [1, 2]:
        return None

    # Evaluate all matrix args
    shapes = []
    for arg in expr.args[1:]:
        try:
            s = evals.eval_expr(unwrap_arg(arg), env, warnings, ctx)
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


def _handle_randi(fname, expr, env, warnings, ctx, evals):
    """randi(imax), randi(imax, n), randi(imax, m, n).

    Like _handle_matrix_constructor but first arg is imax (value range), not a dimension.
    - randi(imax) -> scalar
    - randi(imax, n) -> matrix[n x n]
    - randi(imax, m, n) -> matrix[m x n]
    """
    import analysis.diagnostics as diag

    if len(expr.args) < 1:
        return None

    # Evaluate first arg (imax) for warning propagation, but don't use it for shape
    try:
        _ = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return None

    # Remaining args are dimensions, like _handle_matrix_constructor
    if len(expr.args) == 1:
        return Shape.scalar()
    elif len(expr.args) == 2:
        arg = unwrap_arg(expr.args[1])
        # Check for negative dimension
        dim_interval = evals.get_interval(arg, env, ctx)
        if interval_definitely_negative(dim_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim_interval))
        try:
            d = expr_to_dim_ir(arg, env, ctx)
            return Shape.matrix(d, d)
        except ValueError:
            pass
    elif len(expr.args) == 3:
        arg0 = unwrap_arg(expr.args[1])
        arg1 = unwrap_arg(expr.args[2])
        # Check for negative dimensions
        dim0_interval = evals.get_interval(arg0, env, ctx)
        dim1_interval = evals.get_interval(arg1, env, ctx)
        if interval_definitely_negative(dim0_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim0_interval))
        if interval_definitely_negative(dim1_interval):
            warnings.append(diag.warn_possibly_negative_dim(expr.line, dim1_interval))
        try:
            r = expr_to_dim_ir(arg0, env, ctx)
            c = expr_to_dim_ir(arg1, env, ctx)
            return Shape.matrix(r, c)
        except ValueError:
            pass
    return None


def _handle_fft(fname, expr, env, warnings, ctx, evals):
    """fft(x), ifft(x), fft2(x), ifft2(x) — same shape as input."""
    if len(expr.args) >= 1:
        s = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
        if s.is_scalar() or s.is_matrix():
            return s
    return None


def _handle_sparse_full(fname, expr, env, warnings, ctx, evals):
    """sparse(A) / full(A) — passthrough shape. sparse(m,n) — constructor."""
    # sparse(m, n) — constructor form (check before passthrough)
    if fname == "sparse" and len(expr.args) == 2:
        try:
            r = expr_to_dim_ir(unwrap_arg(expr.args[0]), env, ctx)
            c = expr_to_dim_ir(unwrap_arg(expr.args[1]), env, ctx)
            return Shape.matrix(r, c)
        except ValueError:
            pass
    # sparse(A) / full(A) — passthrough
    if len(expr.args) == 1:
        s = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
        if s.is_scalar() or s.is_matrix():
            return s
    return None


def _handle_cross(fname, expr, env, warnings, ctx, evals):
    """cross(a, b) — returns vector same size as inputs (must be 3-element)."""
    if len(expr.args) >= 1:
        return evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    return None


def _handle_conv(fname, expr, env, warnings, ctx, evals):
    """conv(a, b) — returns vector of length len(a)+len(b)-1."""
    # Conservative: return unknown-length vector
    return Shape.matrix(None, 1)


def _handle_polyfit(fname, expr, env, warnings, ctx, evals):
    """polyfit(x, y, n) — returns row vector of n+1 coefficients."""
    if len(expr.args) >= 3:
        try:
            n_dim = expr_to_dim_ir(unwrap_arg(expr.args[2]), env, ctx)
            return Shape.matrix(1, add_dim(n_dim, 1))
        except ValueError:
            pass
    return Shape.matrix(1, None)


def _handle_polyval(fname, expr, env, warnings, ctx, evals):
    """polyval(p, x) — returns same shape as x."""
    if len(expr.args) >= 2:
        return evals.eval_expr(unwrap_arg(expr.args[1]), env, warnings, ctx)
    return None


def _handle_meshgrid(fname, expr, env, warnings, ctx, evals):
    """meshgrid(x, y) — returns matrix, but we don't track multi-return here."""
    return Shape.matrix(None, None)


def _handle_struct(fname, expr, env, warnings, ctx, evals):
    """struct('field1', val1, 'field2', val2, ...) — infer field names and shapes."""
    # struct() with no args — empty struct
    if len(expr.args) == 0:
        return Shape.struct({})
    # struct('field', val, 'field2', val2, ...) — pairs of string + value
    from ir import StringLit
    fields = {}
    args = expr.args
    i = 0
    while i + 1 < len(args):
        key_expr = unwrap_arg(args[i])
        val_expr = unwrap_arg(args[i + 1])
        if isinstance(key_expr, StringLit):
            val_shape = evals.eval_expr(val_expr, env, warnings, ctx)
            fields[key_expr.value] = val_shape
        else:
            # Non-literal field name — can't track statically
            return Shape.struct({})
        i += 2
    return Shape.struct(fields)


def _handle_fieldnames(fname, expr, env, warnings, ctx, evals):
    """fieldnames(s) — returns cell array of field names."""
    return Shape.cell(None, 1)


def _handle_ndims(fname, expr, env, warnings, ctx, evals):
    """ndims(A) — always returns a scalar."""
    return Shape.scalar()


def _handle_sub2ind(fname, expr, env, warnings, ctx, evals):
    """sub2ind(sz, i, j, ...) — returns same shape as index args."""
    if len(expr.args) >= 2:
        return evals.eval_expr(unwrap_arg(expr.args[1]), env, warnings, ctx)
    return Shape.scalar()


def _handle_horzcat_vertcat(fname, expr, env, warnings, ctx, evals):
    """horzcat(a, b, ...) / vertcat(a, b, ...) — concatenation builtins."""
    if len(expr.args) == 0:
        return Shape.matrix(0, 0)
    shapes = [evals.eval_expr(unwrap_arg(a), env, warnings, ctx) for a in expr.args]
    if fname == "horzcat":
        # Horizontal: rows must match, cols add
        r = shapes[0].rows if shapes[0].is_matrix() else (1 if shapes[0].is_scalar() else None)
        c = 0
        for s in shapes:
            sc = s.cols if s.is_matrix() else (1 if s.is_scalar() else None)
            c = add_dim(c, sc)
        return Shape.matrix(r, c)
    else:  # vertcat
        c = shapes[0].cols if shapes[0].is_matrix() else (1 if shapes[0].is_scalar() else None)
        r = 0
        for s in shapes:
            sr = s.rows if s.is_matrix() else (1 if s.is_scalar() else None)
            r = add_dim(r, sr)
        return Shape.matrix(r, c)


# Dispatch table: builtin name -> handler function (complex builtins only)
# Simple stereotype builtins are handled declaratively via the frozensets above.
BUILTIN_HANDLERS = {
    'zeros': _handle_zeros_ones,
    'ones': _handle_zeros_ones,
    'size': _handle_size,
    'cell': _handle_cell_constructor,
    'transpose': _handle_transpose_fn,
    'reshape': _handle_reshape,
    'repmat': _handle_repmat,
    'diag': _handle_diag,
    'inv': _handle_inv,
    'pinv': _handle_inv,
    'linspace': _handle_linspace,
    'logspace': _handle_linspace,
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
    'randi': _handle_randi,
    'find': _handle_find,
    'cat': _handle_cat,
    'eig': _handle_eig_single,
    'svd': _handle_svd_single,
    'fft': _handle_fft,
    'ifft': _handle_fft,
    'fft2': _handle_fft,
    'ifft2': _handle_fft,
    'sparse': _handle_sparse_full,
    'full': _handle_sparse_full,
    'cross': _handle_cross,
    'conv': _handle_conv,
    'deconv': _handle_conv,
    'polyfit': _handle_polyfit,
    'polyval': _handle_polyval,
    'interp1': _handle_polyval,
    'meshgrid': _handle_meshgrid,
    'struct': _handle_struct,
    'fieldnames': _handle_fieldnames,
    'ndims': _handle_ndims,
    'sub2ind': _handle_sub2ind,
    'horzcat': _handle_horzcat_vertcat,
    'vertcat': _handle_horzcat_vertcat,
}

# Verify no overlap between declarative sets and BUILTIN_HANDLERS
assert not (_all_declarative & set(BUILTIN_HANDLERS)), \
    f"Overlap between declarative sets and BUILTIN_HANDLERS: {_all_declarative & set(BUILTIN_HANDLERS)}"



def _eval_first_arg_shape(expr, env, warnings, ctx, evals):
    """Evaluate first arg and return (rows, cols) or (None, None) for non-matrix."""
    if len(expr.args) < 1:
        return None, None
    try:
        s = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return None, None
    if s.is_scalar():
        return 1, 1
    if s.is_matrix():
        return s.rows, s.cols
    return None, None


def _handle_multi_eig(fname, expr, env, warnings, ctx, num_targets, evals):
    """[V, D] = eig(A) -> [matrix[n x n], matrix[n x n]]."""
    if num_targets != 2:
        return None
    r, c = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    if r is None:
        return [Shape.unknown(), Shape.unknown()]
    # Use r for both dimensions (square matrix assumption)
    n = join_dim(r, c) if r != c else r
    return [Shape.matrix(n, n), Shape.matrix(n, n)]


def _handle_multi_svd(fname, expr, env, warnings, ctx, num_targets, evals):
    """[U, S, V] = svd(A) -> [matrix[m x m], matrix[m x n], matrix[n x n]]."""
    if num_targets != 3:
        return None
    m, n = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    if m is None:
        return [Shape.unknown(), Shape.unknown(), Shape.unknown()]
    return [Shape.matrix(m, m), Shape.matrix(m, n), Shape.matrix(n, n)]


def _handle_multi_lu(fname, expr, env, warnings, ctx, num_targets, evals):
    """[L, U] = lu(A) or [L, U, P] = lu(A)."""
    m, n = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    if m is None:
        if num_targets == 2:
            return [Shape.unknown(), Shape.unknown()]
        elif num_targets == 3:
            return [Shape.unknown(), Shape.unknown(), Shape.unknown()]
        return None
    if num_targets == 2:
        # 2-output: [L, U] = [m x m, m x n]
        return [Shape.matrix(m, m), Shape.matrix(m, n)]
    elif num_targets == 3:
        # 3-output: [L, U, P] = [m x n, n x n, m x m]
        return [Shape.matrix(m, n), Shape.matrix(n, n), Shape.matrix(m, m)]
    return None


def _handle_multi_qr(fname, expr, env, warnings, ctx, num_targets, evals):
    """[Q, R] = qr(A) -> [matrix[m x m], matrix[m x n]]."""
    if num_targets != 2:
        return None
    m, n = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    if m is None:
        return [Shape.unknown(), Shape.unknown()]
    return [Shape.matrix(m, m), Shape.matrix(m, n)]


def _handle_multi_chol(fname, expr, env, warnings, ctx, num_targets, evals):
    """[R, p] = chol(A) -> [matrix[n x n], scalar]."""
    if num_targets != 2:
        return None
    r, c = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    if r is None:
        return [Shape.unknown(), Shape.scalar()]
    # Use r for both dimensions (square matrix assumption)
    n = join_dim(r, c) if r != c else r
    return [Shape.matrix(n, n), Shape.scalar()]


def _handle_multi_size(fname, expr, env, warnings, ctx, num_targets, evals):
    """[m, n] = size(A) -> [scalar, scalar]."""
    if num_targets != 2:
        return None
    # Evaluate first arg for warning propagation
    _, _ = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
    return [Shape.scalar(), Shape.scalar()]


def _handle_multi_sort(fname, expr, env, warnings, ctx, num_targets, evals):
    """[s, i] = sort(A) -> [shape_of(A), shape_of(A)]."""
    if num_targets != 2:
        return None
    if len(expr.args) < 1:
        return [Shape.unknown(), Shape.unknown()]
    try:
        s = evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
    except ValueError:
        return [Shape.unknown(), Shape.unknown()]
    return [s, s]


def _handle_multi_find(fname, expr, env, warnings, ctx, num_targets, evals):
    """[row, col] = find(A) or [row, col, val] = find(A)."""
    if num_targets == 2:
        _, _ = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
        return [Shape.matrix(1, None), Shape.matrix(1, None)]
    elif num_targets == 3:
        _, _ = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
        return [Shape.matrix(1, None), Shape.matrix(1, None), Shape.matrix(1, None)]
    return None


def _handle_multi_unique(fname, expr, env, warnings, ctx, num_targets, evals):
    """[u, ia] = unique(A) or [u, ia, ic] = unique(A)."""
    if num_targets == 2:
        _, _ = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
        return [Shape.matrix(1, None), Shape.matrix(None, 1)]
    elif num_targets == 3:
        _, _ = _eval_first_arg_shape(expr, env, warnings, ctx, evals)
        return [Shape.matrix(1, None), Shape.matrix(None, 1), Shape.matrix(None, 1)]
    return None


def _handle_multi_minmax(fname, expr, env, warnings, ctx, num_targets, evals):
    """[M, I] = min(A) or [M, I] = max(A)."""
    if num_targets != 2:
        return None
    if len(expr.args) < 1:
        return [Shape.unknown(), Shape.unknown()]
    arg_shape = evals.eval_arg(expr.args[0], env, warnings, ctx)
    # Apply reduction logic (same as _handle_reduction for 1-arg)
    if arg_shape.is_scalar():
        return [Shape.scalar(), Shape.scalar()]
    if arg_shape.is_matrix():
        reduction = Shape.matrix(1, arg_shape.cols)
        return [reduction, reduction]
    return [Shape.unknown(), Shape.unknown()]


def _handle_multi_fileparts(fname, expr, env, warnings, ctx, num_targets, evals):
    """[pathstr, name, ext] = fileparts(filename) — all strings."""
    if num_targets <= 3:
        return [Shape.string()] * num_targets
    return None


def _handle_multi_fopen(fname, expr, env, warnings, ctx, num_targets, evals):
    """[fid, message] = fopen(filename) — scalar + string."""
    if num_targets == 2:
        return [Shape.scalar(), Shape.string()]
    return None


def _handle_multi_meshgrid(fname, expr, env, warnings, ctx, num_targets, evals):
    """[X, Y] = meshgrid(x, y) or [X, Y, Z] = meshgrid(x, y, z) — matrices."""
    if num_targets in (2, 3):
        return [Shape.unknown()] * num_targets
    return None


def _handle_multi_cellfun(fname, expr, env, warnings, ctx, num_targets, evals):
    """[out1, out2, ...] = cellfun(func, C) — all unknown."""
    return [Shape.unknown()] * num_targets


def _handle_multi_ndgrid(fname, expr, env, warnings, ctx, num_targets, evals):
    """[X1, X2, ...] = ndgrid(x1, x2, ...) — matrices."""
    return [Shape.unknown()] * num_targets


def _handle_multi_regexp(fname, expr, env, warnings, ctx, num_targets, evals):
    """[tok, match, ...] = regexp(str, expr, ...) — all unknown."""
    return [Shape.unknown()] * num_targets


# Supported forms lookup for count mismatch messages
_MULTI_SUPPORTED_FORMS = {
    'eig': '1 or 2', 'svd': '1 or 3', 'lu': '2 or 3', 'qr': '2',
    'chol': '2', 'size': '2', 'sort': '2', 'find': '1, 2, or 3',
    'unique': '1, 2, or 3', 'min': '1 or 2', 'max': '1 or 2',
    'fileparts': '1-3', 'fopen': '2', 'meshgrid': '2 or 3',
    'cellfun': 'any', 'ndgrid': 'any', 'regexp': 'any', 'regexpi': 'any',
}


# Multi-return builtin dispatch table
BUILTIN_MULTI_HANDLERS = {
    'eig': _handle_multi_eig,
    'svd': _handle_multi_svd,
    'lu': _handle_multi_lu,
    'qr': _handle_multi_qr,
    'chol': _handle_multi_chol,
    'size': _handle_multi_size,
    'sort': _handle_multi_sort,
    'find': _handle_multi_find,
    'unique': _handle_multi_unique,
    'min': _handle_multi_minmax,
    'max': _handle_multi_minmax,
    'fileparts': _handle_multi_fileparts,
    'fopen': _handle_multi_fopen,
    'meshgrid': _handle_multi_meshgrid,
    'cellfun': _handle_multi_cellfun,
    'ndgrid': _handle_multi_ndgrid,
    'regexp': _handle_multi_regexp,
    'regexpi': _handle_multi_regexp,
}

# Verify all handled builtins are in KNOWN_BUILTINS (prevents orphan handlers)
from analysis.builtins import KNOWN_BUILTINS as _KNOWN
_all_handled = _all_declarative | set(BUILTIN_HANDLERS) | set(BUILTIN_MULTI_HANDLERS)
_orphans = _all_handled - _KNOWN
assert not _orphans, f"Builtins with handlers but not in KNOWN_BUILTINS: {_orphans}"


def eval_builtin_call(fname: str, expr: Apply, env: Env, warnings: List['Diagnostic'], ctx: AnalysisContext, evals: BuiltinEvalContext) -> Shape:
    """Dispatch builtin function call to appropriate handler.

    Args:
        fname: Builtin function name
        expr: Apply expression node
        env: Current environment
        warnings: List to append warnings to
        ctx: Analysis context
        evals: Evaluation callbacks (breaks circular import)

    Returns:
        Inferred shape of the result
    """
    # Complex handlers take priority (checked first)
    handler = BUILTIN_HANDLERS.get(fname)
    if handler:
        try:
            result = handler(fname, expr, env, warnings, ctx, evals)
        except ValueError:
            # Range/Colon args passed to builtins expecting plain Expr
            return Shape.unknown()
        if result is not None:
            return result

    # Declarative dispatch for stereotyped builtins
    if fname in PASSTHROUGH_BUILTINS or fname in TYPE_CAST_BUILTINS:
        # Calls eval_arg (handles IndexArg/Colon/Range directly)
        # Only handles the 1-arg case; 2+ args fall through to unknown
        if len(expr.args) == 1:
            return evals.eval_arg(expr.args[0], env, warnings, ctx)
        return Shape.unknown()

    if fname in SCALAR_PREDICATE_BUILTINS:
        # Calls eval_expr(unwrap_arg(...)) — unwraps first, then evaluates
        if len(expr.args) >= 1:
            try:
                evals.eval_expr(unwrap_arg(expr.args[0]), env, warnings, ctx)
            except ValueError:
                pass
        return Shape.scalar()

    if fname in SCALAR_QUERY_BUILTINS:
        # Calls eval_arg (like passthrough), returns scalar
        if len(expr.args) >= 1:
            _ = evals.eval_arg(expr.args[0], env, warnings, ctx)
        return Shape.scalar()

    if fname in SCALAR_NARY_BUILTINS:
        # No argument evaluation — always returns scalar
        return Shape.scalar()

    if fname in STRING_RETURN_BUILTINS:
        # Evaluate ALL args for warning propagation, return string
        for arg in expr.args:
            _ = evals.eval_arg(arg, env, warnings, ctx)
        return Shape.string()

    if fname in REDUCTION_BUILTINS:
        # Delegate to existing _handle_reduction logic
        return _handle_reduction(fname, expr, env, warnings, ctx, evals) or Shape.unknown()

    if fname in MATRIX_CONSTRUCTOR_BUILTINS:
        # Delegate to existing _handle_matrix_constructor logic
        return _handle_matrix_constructor(fname, expr, env, warnings, ctx, evals) or Shape.unknown()

    # Known builtin without a matching shape rule: return unknown silently
    return Shape.unknown()
