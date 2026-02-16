# Ethan Doughty
# eval_expr.py
"""Expression evaluation — shape inference for IR expressions."""

from __future__ import annotations
from typing import List, Optional

from analysis.builtins import KNOWN_BUILTINS
from analysis.context import AnalysisContext
from analysis.end_helpers import _binop_contains_end, _eval_end_arithmetic
from analysis.dim_extract import expr_to_dim_ir, expr_to_dim_ir_with_end
from analysis.eval_binop import eval_binop_ir
from analysis.eval_builtins import eval_builtin_call
from analysis.func_analysis import analyze_function_call

from ir import (
    Expr, Var, Const, StringLit, FieldAccess, Lambda, FuncHandle, End,
    MatrixLit, CellLit, Apply, CurlyApply, Transpose, Neg, BinOp,
    IndexArg, Colon, Range, IndexExpr,
)

import analysis.diagnostics as diag
from runtime.env import Env
from runtime.shapes import Shape, Dim, join_shape
from analysis.matrix_literals import infer_matrix_literal_shape


def _eval_index_arg_to_shape(arg: IndexArg, env: Env, warnings: List[str], ctx: AnalysisContext, container_shape: Optional[Shape] = None) -> Shape:
    """Evaluate an IndexArg to a Shape.

    Handles:
    - IndexExpr: unwrap and evaluate the inner expression
    - Range: evaluate as a row vector (1 x n)
    - Colon: cannot be evaluated standalone, return unknown

    Args:
        arg: IndexArg to evaluate
        env: Current environment
        warnings: List to append warnings to
        ctx: Analysis context
        container_shape: Optional shape of container being indexed (for End resolution)

    Returns:
        Shape of the argument
    """
    if isinstance(arg, IndexExpr):
        return eval_expr_ir(arg.expr, env, warnings, ctx, container_shape)
    elif isinstance(arg, Range):
        # Range creates a row vector; shape depends on the bounds
        # Evaluate start and end for side effects (e.g., End keyword resolution)
        _ = eval_expr_ir(arg.start, env, warnings, ctx, container_shape)
        _ = eval_expr_ir(arg.end, env, warnings, ctx, container_shape)
        # For simplicity, we assume ranges create 1 x N matrices
        # (The actual size depends on start/end evaluation)
        return Shape.matrix(1, None)  # 1 x unknown
    elif isinstance(arg, Colon):
        # Standalone colon cannot be evaluated
        return Shape.unknown()
    return Shape.unknown()


def eval_expr_ir(expr: Expr, env: Env, warnings: List[str], ctx: AnalysisContext, container_shape: Optional[Shape] = None) -> Shape:
    """Evaluate an expression to infer its shape.

    Args:
        expr: Expression to evaluate
        env: Current environment with variable shapes
        warnings: List to append warnings to
        ctx: Analysis context
        container_shape: Optional shape of container being indexed (for End keyword resolution)

    Returns:
        Inferred shape of the expression
    """
    # Variables / constants
    if isinstance(expr, Var):
        shape = env.get(expr.name)
        # Convert bottom -> unknown at expression evaluation boundary
        # (bottom is internal to Env, expression eval never sees it)
        return shape if not shape.is_bottom() else Shape.unknown()

    if isinstance(expr, Const):
        return Shape.scalar()

    if isinstance(expr, StringLit):
        return Shape.string()

    if isinstance(expr, End):
        # End keyword in indexing context
        if container_shape is None:
            warnings.append(diag.warn_end_outside_indexing(expr.line))
            return Shape.unknown()
        # End resolves to a scalar index value (integer)
        return Shape.scalar()

    if isinstance(expr, MatrixLit):
        # Evaluate element shapes first (before conversion)
        raw_shape_rows = [
            [eval_expr_ir(e, env, warnings, ctx) for e in row]
            for row in expr.rows
        ]
        return infer_matrix_literal_shape(raw_shape_rows, expr.line, warnings)

    if isinstance(expr, CellLit):
        # Cell literal: infer dimensions and track element shapes
        if not expr.rows:
            return Shape.cell(0, 0, elements={})

        num_rows = len(expr.rows)
        num_cols = len(expr.rows[0]) if expr.rows else 0

        # Validate rectangular structure (all rows same length)
        for row in expr.rows:
            if len(row) != num_cols:
                # Ragged cell array: return unknown dimensions, no element tracking
                return Shape.cell(None, None, elements=None)

        # Evaluate all elements and collect shapes (COLUMN-MAJOR indexing)
        elem_shapes = {}
        for row_idx, row_exprs in enumerate(expr.rows):
            for col_idx, e in enumerate(row_exprs):
                elem_shape = eval_expr_ir(e, env, warnings, ctx)
                # Column-major: linear_idx = col * num_rows + row
                linear_idx = col_idx * num_rows + row_idx
                # Only store non-unknown shapes (sparse representation)
                if not elem_shape.is_unknown():
                    elem_shapes[linear_idx] = elem_shape

        return Shape.cell(num_rows, num_cols, elements=elem_shapes)

    if isinstance(expr, CurlyApply):
        # Curly indexing: c{i} or c{i,j}
        base_shape = eval_expr_ir(expr.base, env, warnings, ctx)

        # Check base is cell
        if not base_shape.is_cell():
            # Evaluate args for side effects
            for arg in expr.args:
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
            warnings.append(diag.warn_curly_indexing_non_cell(expr.line, base_shape))
            return Shape.unknown()

        # If no element tracking, return unknown
        if base_shape._elements is None:
            # Evaluate args for side effects
            for arg in expr.args:
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
            return Shape.unknown()

        elem_dict = dict(base_shape._elements)

        # Try to extract literal index
        if len(expr.args) == 1:
            # 1D indexing: c{i}
            arg = expr.args[0]
            if isinstance(arg, IndexExpr) and isinstance(arg.expr, Const):
                # Literal 1D index
                idx_value = int(arg.expr.value) - 1  # Convert MATLAB 1-based to 0-based
                return elem_dict.get(idx_value, Shape.unknown())
            elif isinstance(arg, IndexExpr) and isinstance(arg.expr, End):
                # c{end} — resolve to last element (linear indexing)
                if isinstance(base_shape.rows, int) and isinstance(base_shape.cols, int):
                    # Concrete dimensions: compute total elements
                    total_elems = base_shape.rows * base_shape.cols
                    idx_value = total_elems - 1  # Convert to 0-based
                    return elem_dict.get(idx_value, Shape.unknown())
                else:
                    # Symbolic/unknown dimensions: can't resolve end statically
                    # Join all elements conservatively
                    if not elem_dict:
                        return Shape.unknown()
                    result = Shape.bottom()
                    for elem_shape in elem_dict.values():
                        result = join_shape(result, elem_shape)
                    return result
            elif isinstance(arg, IndexExpr) and isinstance(arg.expr, BinOp) and _binop_contains_end(arg.expr):
                # c{end-1}, c{end/2}, etc. — resolve end arithmetic
                if isinstance(base_shape.rows, int) and isinstance(base_shape.cols, int):
                    # Concrete dimensions: compute total elements
                    total_elems = base_shape.rows * base_shape.cols
                    idx_value = _eval_end_arithmetic(arg.expr, total_elems)
                    if idx_value is not None and 1 <= idx_value <= total_elems:
                        # Valid 1-based index, convert to 0-based
                        return elem_dict.get(idx_value - 1, Shape.unknown())
                    else:
                        # Out of bounds or can't resolve (contains variables) → join all elements conservatively
                        if not elem_dict:
                            return Shape.unknown()
                        result = Shape.bottom()
                        for elem_shape in elem_dict.values():
                            result = join_shape(result, elem_shape)
                        return result
                else:
                    # Symbolic/unknown dimensions: can't resolve end statically
                    # Join all elements conservatively
                    if not elem_dict:
                        return Shape.unknown()
                    result = Shape.bottom()
                    for elem_shape in elem_dict.values():
                        result = join_shape(result, elem_shape)
                    return result
            elif isinstance(arg, Colon):
                # Colon indexing: c{:} → join all elements
                if not elem_dict:
                    return Shape.unknown()
                result = Shape.bottom()
                for elem_shape in elem_dict.values():
                    result = join_shape(result, elem_shape)
                return result
            elif isinstance(arg, Range):
                # Range indexing: c{start:end} → join subset
                # For now, conservatively join all elements (precise range extraction deferred)
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
                if not elem_dict:
                    return Shape.unknown()
                result = Shape.bottom()
                for elem_shape in elem_dict.values():
                    result = join_shape(result, elem_shape)
                return result
            else:
                # Dynamic index: join all elements
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
                if not elem_dict:
                    return Shape.unknown()
                result = Shape.bottom()
                for elem_shape in elem_dict.values():
                    result = join_shape(result, elem_shape)
                return result

        elif len(expr.args) == 2:
            # 2D indexing: c{i, j}
            arg_row = expr.args[0]
            arg_col = expr.args[1]

            # Try to extract concrete row and column indices (handle both Const and End)
            row_idx = None
            col_idx = None

            if isinstance(arg_row, IndexExpr):
                if isinstance(arg_row.expr, Const):
                    row_idx = int(arg_row.expr.value) - 1  # Convert to 0-based
                elif isinstance(arg_row.expr, End):
                    # end in row position → last row
                    if isinstance(base_shape.rows, int):
                        row_idx = base_shape.rows - 1
                    # else: symbolic/unknown, leave as None
                elif isinstance(arg_row.expr, BinOp) and _binop_contains_end(arg_row.expr):
                    # end-1, end/2, etc. in row position
                    if isinstance(base_shape.rows, int):
                        idx_value = _eval_end_arithmetic(arg_row.expr, base_shape.rows)
                        if idx_value is not None and 1 <= idx_value <= base_shape.rows:
                            row_idx = idx_value - 1  # Convert to 0-based
                        # else: out of bounds or can't resolve, leave as None

            if isinstance(arg_col, IndexExpr):
                if isinstance(arg_col.expr, Const):
                    col_idx = int(arg_col.expr.value) - 1  # Convert to 0-based
                elif isinstance(arg_col.expr, End):
                    # end in col position → last col
                    if isinstance(base_shape.cols, int):
                        col_idx = base_shape.cols - 1
                    # else: symbolic/unknown, leave as None
                elif isinstance(arg_col.expr, BinOp) and _binop_contains_end(arg_col.expr):
                    # end-1, end/2, etc. in col position
                    if isinstance(base_shape.cols, int):
                        idx_value = _eval_end_arithmetic(arg_col.expr, base_shape.cols)
                        if idx_value is not None and 1 <= idx_value <= base_shape.cols:
                            col_idx = idx_value - 1  # Convert to 0-based
                        # else: out of bounds or can't resolve, leave as None

            if row_idx is not None and col_idx is not None and isinstance(base_shape.rows, int):
                # Both indices resolved: compute linear index
                linear_idx = col_idx * base_shape.rows + row_idx
                return elem_dict.get(linear_idx, Shape.unknown())
            else:
                # Dynamic 2D index: evaluate args and join all elements
                _ = _eval_index_arg_to_shape(arg_row, env, warnings, ctx, container_shape=base_shape)
                _ = _eval_index_arg_to_shape(arg_col, env, warnings, ctx, container_shape=base_shape)
                if not elem_dict:
                    return Shape.unknown()
                result = Shape.bottom()
                for elem_shape in elem_dict.values():
                    result = join_shape(result, elem_shape)
                return result

        else:
            # Multi-dimensional indexing (>2D): evaluate args and return unknown
            for arg in expr.args:
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
            return Shape.unknown()

    if isinstance(expr, Apply):
        line = expr.line

        # Check for indexing indicators: colon or range in args
        has_colon_or_range = any(isinstance(arg, (Colon, Range)) for arg in expr.args)

        # Check if base variable is a function handle (shadows builtins)
        if isinstance(expr.base, Var):
            base_var_name = expr.base.name
            base_var_shape = env.get(base_var_name)
            if base_var_shape.is_function_handle():
                # Function handle call: analyze lambda/handle bodies
                if base_var_shape._lambda_ids is None:
                    # Opaque handle (no IDs) → fallback to approximate
                    warnings.append(diag.warn_lambda_call_approximate(line, base_var_name))
                    return Shape.unknown()

                # Evaluate all argument shapes
                arg_shapes = []
                for arg in expr.args:
                    if isinstance(arg, (Colon, Range)):
                        # Colon/Range in function call args is invalid
                        arg_shapes.append(Shape.unknown())
                    elif isinstance(arg, IndexExpr):
                        arg_shape = eval_expr_ir(arg.expr, env, warnings, ctx)
                        arg_shapes.append(arg_shape)
                    else:
                        # Should not happen (Apply.args are IndexArg)
                        arg_shapes.append(Shape.unknown())

                # Analyze all possible callables (lambdas + handles)
                results = []
                for callable_id in sorted(base_var_shape._lambda_ids):  # sorted for determinism
                    if callable_id in ctx._lambda_metadata:
                        # Lambda body analysis
                        params, body_expr, closure_env = ctx._lambda_metadata[callable_id]
                        # Compute dimension aliases for cache key
                        arg_dim_aliases = tuple(
                            (param, expr_to_dim_ir(arg.expr, env)) if isinstance(arg, IndexExpr) else (param, None)
                            for param, arg in zip(params, expr.args)
                        )
                        cache_key = ("lambda", callable_id, tuple(arg_shapes), arg_dim_aliases)

                        # Cache check
                        if cache_key in ctx.analysis_cache:
                            cached_shape, cached_warnings = ctx.analysis_cache[cache_key]
                            warnings.extend(cached_warnings)
                            results.append(cached_shape)
                            continue

                        # Recursion guard
                        if callable_id in ctx.analyzing_lambdas:
                            warnings.append(diag.warn_recursive_lambda(line))
                            results.append(Shape.unknown())
                            continue

                        # Check arg count
                        if len(arg_shapes) != len(params):
                            warnings.append(diag.warn_lambda_arg_count_mismatch(line, len(params), len(arg_shapes)))
                            results.append(Shape.unknown())
                            continue

                        # Analyze lambda body
                        ctx.analyzing_lambdas.add(callable_id)
                        try:
                            # Create env from closure snapshot + bind params
                            call_env = closure_env.copy()
                            # Allow self-reference for recursion detection
                            # (enables f = @(x) f(x-1) to trigger recursion guard)
                            call_env.set(base_var_name, Shape.function_handle(lambda_ids=frozenset({callable_id})))
                            for i, (param, arg_shape) in enumerate(zip(params, arg_shapes)):
                                call_env.set(param, arg_shape)
                                # Dimension aliasing: extract dimension from arg expression
                                arg = expr.args[i]
                                if isinstance(arg, IndexExpr):
                                    caller_dim = expr_to_dim_ir(arg.expr, env)
                                    if caller_dim is not None:
                                        call_env.dim_aliases[param] = caller_dim

                            # Analyze body expression
                            lambda_warnings = []
                            result = eval_expr_ir(body_expr, call_env, lambda_warnings, ctx)

                            # Cache result
                            ctx.analysis_cache[cache_key] = (result, lambda_warnings)
                            warnings.extend(lambda_warnings)
                            results.append(result)
                        finally:
                            ctx.analyzing_lambdas.discard(callable_id)

                    elif callable_id in ctx._handle_registry:
                        # Function handle dispatch (Phase 3)
                        func_name = ctx._handle_registry[callable_id]
                        if func_name in ctx.function_registry:
                            # Dispatch to user-defined function
                            output_shapes = analyze_function_call(func_name, expr.args, line, env, warnings, ctx)
                            # Use first output (or unknown if none)
                            if len(output_shapes) >= 1:
                                results.append(output_shapes[0])
                            else:
                                results.append(Shape.unknown())
                        elif func_name in KNOWN_BUILTINS:
                            # Dispatch to builtin: create synthetic Apply and recursively evaluate
                            # Build a synthetic Var with the builtin name (not the handle variable name)
                            # to avoid infinite recursion
                            synthetic_apply = Apply(
                                base=Var(name=func_name, line=line),
                                args=expr.args,
                                line=line
                            )
                            # Recursively evaluate — will dispatch to builtin logic
                            # Safe because the synthetic Var is not bound as a function_handle
                            builtin_result = eval_expr_ir(synthetic_apply, env, warnings, ctx)
                            results.append(builtin_result)
                        else:
                            # Function not found (should not happen — validated at FuncHandle eval)
                            results.append(Shape.unknown())

                # Join all results
                if not results:
                    # No lambda metadata or handle registry (should not happen)
                    warnings.append(diag.warn_lambda_call_approximate(line, base_var_name))
                    return Shape.unknown()

                joined = results[0]
                for r in results[1:]:
                    joined = join_shape(joined, r)
                return joined

        # Check if base is a known builtin function
        if isinstance(expr.base, Var):
            fname = expr.base.name
            if fname in KNOWN_BUILTINS:
                return eval_builtin_call(fname, expr, env, warnings, ctx)

            # Check if variable is unbound (unknown function)
            if fname not in env.bindings:
                # Check function registry before giving up
                if fname in ctx.function_registry:
                    output_shapes = analyze_function_call(fname, expr.args, line, env, warnings, ctx)

                    # Check if procedure (no return value)
                    if len(ctx.function_registry[fname].output_vars) == 0:
                        warnings.append(diag.warn_procedure_in_expr(line, fname))
                        return Shape.unknown()

                    if len(output_shapes) == 1:
                        return output_shapes[0]
                    else:
                        # Multiple returns in expression context, use first return value
                        return output_shapes[0]

                # Truly unknown function
                warnings.append(diag.warn_unknown_function(line, fname))
                return Shape.unknown()

        # Default: treat as indexing (bound variable)
        base_shape = eval_expr_ir(expr.base, env, warnings, ctx)
        return _eval_indexing(base_shape, expr.args, line, expr, env, warnings, ctx)

    if isinstance(expr, Transpose):
        inner = eval_expr_ir(expr.operand, env, warnings, ctx)
        if inner.is_matrix():
            return Shape.matrix(inner.cols, inner.rows)
        return inner

    if isinstance(expr, Neg):
        return eval_expr_ir(expr.operand, env, warnings, ctx)

    if isinstance(expr, FieldAccess):
        # Struct field access: s.field or s.a.b (nested)
        base_shape = eval_expr_ir(expr.base, env, warnings, ctx)
        if base_shape.is_struct():
            field_shape = base_shape.fields_dict.get(expr.field)
            if field_shape is None:
                # Field not found in struct
                warnings.append(diag.warn_struct_field_not_found(expr.line, expr.field, base_shape))
                return Shape.unknown()
            # Convert bottom → unknown at expression boundary
            return field_shape if not field_shape.is_bottom() else Shape.unknown()
        else:
            # Base is not a struct
            warnings.append(diag.warn_field_access_non_struct(expr.line, base_shape))
            return Shape.unknown()

    if isinstance(expr, Lambda):
        # Anonymous function: store closure and return function_handle
        # Increment lambda ID counter
        lambda_id = ctx._next_lambda_id
        ctx._next_lambda_id += 1
        # Store snapshot of current environment as closure (FIX: env.copy())
        ctx._lambda_metadata[lambda_id] = (expr.params, expr.body, env.copy())
        return Shape.function_handle(lambda_ids=frozenset({lambda_id}))

    if isinstance(expr, FuncHandle):
        # Named function handle: check if function exists
        handle_id = ctx._next_lambda_id
        ctx._next_lambda_id += 1
        if expr.name in ctx.function_registry or expr.name in KNOWN_BUILTINS:
            ctx._handle_registry[handle_id] = expr.name
            return Shape.function_handle(lambda_ids=frozenset({handle_id}))
        else:
            # Unknown function
            warnings.append(diag.warn_unknown_function(expr.line, expr.name))
            return Shape.function_handle()  # Opaque (no IDs)

    if isinstance(expr, BinOp):
        op = expr.op
        line = expr.line
        left_shape = eval_expr_ir(expr.left, env, warnings, ctx, container_shape)
        right_shape = eval_expr_ir(expr.right, env, warnings, ctx, container_shape)
        return eval_binop_ir(op, left_shape, right_shape, warnings, expr.left, expr.right, line)

    return Shape.unknown()

def _eval_indexing(base_shape: Shape, args, line: int, expr, env: Env, warnings: List[str], ctx: AnalysisContext) -> Shape:
    """Indexing logic for Apply-as-indexing nodes.

    Args:
        base_shape: Shape of the base expression being indexed
        args: List of IndexArg arguments
        line: Source line number
        expr: Original expression (for diagnostics)
        env: Current environment
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        Inferred shape of the indexing result
    """
    if base_shape.is_unknown():
        return Shape.unknown()

    if base_shape.is_scalar():
        warnings.append(diag.warn_indexing_scalar(line, expr))
        return Shape.unknown()

    if base_shape.is_matrix():
        m = base_shape.rows
        n = base_shape.cols

        if len(args) == 1:
            return Shape.scalar()

        if len(args) == 2:
            a1, a2 = args

            r_extent = index_arg_to_extent_ir(a1, env, warnings, line, ctx, container_shape=base_shape)
            c_extent = index_arg_to_extent_ir(a2, env, warnings, line, ctx, container_shape=base_shape)

            def is_allowed_unknown(a: IndexArg) -> bool:
                return isinstance(a, (Colon, Range))

            if (r_extent is None and not is_allowed_unknown(a1)) or (c_extent is None and not is_allowed_unknown(a2)):
                return Shape.unknown()

            if isinstance(a1, Colon):
                r_extent = m
            if isinstance(a2, Colon):
                c_extent = n

            if r_extent == 1 and c_extent == 1:
                return Shape.scalar()

            return Shape.matrix(r_extent, c_extent)

        if len(args) > 2:
            warnings.append(diag.warn_too_many_indices(line, expr))
            return Shape.unknown()

    return Shape.unknown()


def index_arg_to_extent_ir(
    arg: IndexArg,
    env: Env,
    warnings: List[str],
    line: int,
    ctx: AnalysisContext,
    container_shape: Optional[Shape] = None
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

        start_shape = eval_expr_ir(start_expr, env, warnings, ctx, container_shape)
        end_shape = eval_expr_ir(end_expr, env, warnings, ctx, container_shape)

        if start_shape.is_matrix() or end_shape.is_matrix():
            warnings.append(diag.warn_range_endpoints_must_be_scalar(line, arg, start_shape, end_shape))
            return None

        # Try to extract concrete endpoints (with End resolution if present)
        a = expr_to_dim_ir(start_expr, env)
        b = expr_to_dim_ir(end_expr, env)

        # Special handling for End in range endpoints
        if (a is None or b is None) and container_shape is not None:
            if container_shape.is_matrix():
                container_dim = None
                if isinstance(container_shape.rows, int):
                    container_dim = container_shape.rows
                elif isinstance(container_shape.cols, int):
                    container_dim = container_shape.cols

                if container_dim is not None:
                    if a is None:
                        a = expr_to_dim_ir_with_end(start_expr, env, container_dim)
                    if b is None:
                        b = expr_to_dim_ir_with_end(end_expr, env, container_dim)

        if isinstance(a, int) and isinstance(b, int):
            if b < a:
                warnings.append(diag.warn_invalid_range_end_lt_start(line, arg))
                return None
            return (b - a) + 1

        return None

    if isinstance(arg, IndexExpr):
        s = eval_expr_ir(arg.expr, env, warnings, ctx, container_shape)
        if s.is_matrix():
            warnings.append(diag.warn_non_scalar_index_arg(line, arg, s))
            return None
        return 1

    return None
