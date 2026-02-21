# Ethan Doughty
# func_analysis.py
"""Function call analysis, loop body analysis, and warning formatting."""

from __future__ import annotations
from typing import List, TYPE_CHECKING

from ir import IndexArg, IndexExpr

import analysis.diagnostics as diag
from analysis.context import FunctionSignature, EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.dim_extract import expr_to_dim_ir
from runtime.env import Env, widen_env
from runtime.shapes import Shape

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic
    from analysis.workspace import ExternalSignature


def _format_dual_location_warning(func_warn: 'Diagnostic', func_name: str, call_line: int) -> 'Diagnostic':
    """Reformat a function-internal diagnostic with call-site context.

    Args:
        func_warn: Diagnostic from function body
        func_name: Name of the called function
        call_line: Line number of the call site

    Returns:
        Diagnostic with dual-location context
    """
    # Check if already has call context
    if "(in " in func_warn.message:
        # Already has call context, return as-is
        return func_warn

    # Create new diagnostic with augmented message
    augmented_message = f"{func_warn.message} (in {func_name}, called from line {call_line})"

    return diag.Diagnostic(
        line=func_warn.line,
        code=func_warn.code,
        message=augmented_message,
        related_line=call_line
    )


def analyze_function_call(
    func_name: str,
    args: List[IndexArg],
    line: int,
    env: Env,
    warnings: List['Diagnostic'],
    ctx: AnalysisContext
) -> List[Shape]:
    """Analyze user-defined function call and return output shapes.

    Uses polymorphic caching keyed by (func_name, arg_shapes tuple).

    Args:
        func_name: Name of function to call
        args: Argument list from Apply node
        line: Call site line number
        env: Caller's environment
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        List of output shapes (one per output_var in function signature)
    """
    from analysis.eval_expr import eval_expr_ir, _eval_index_arg_to_shape
    from analysis.stmt_analysis import analyze_stmt_ir

    if func_name not in ctx.call.function_registry:
        # Should not reach here (checked by caller)
        return [Shape.unknown()]

    sig = ctx.call.function_registry[func_name]

    # Check argument count
    if len(args) != len(sig.params):
        warnings.append(diag.warn_function_arg_count_mismatch(
            line, func_name, expected=len(sig.params), got=len(args)
        ))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Recursion guard
    if func_name in ctx.call.analyzing_functions:
        warnings.append(diag.warn_recursive_function(line, func_name))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Evaluate arg shapes for cache key
    arg_shapes = tuple(_eval_index_arg_to_shape(arg, env, warnings, ctx) for arg in args)
    # Compute dimension aliases for cache key (symbolic var names differ per call site)
    arg_dim_aliases = tuple(
        (param, expr_to_dim_ir(arg.expr, env)) if isinstance(arg, IndexExpr) else (param, None)
        for param, arg in zip(sig.params, args)
    )
    cache_key = (func_name, arg_shapes, arg_dim_aliases)

    # Check cache
    if cache_key in ctx.call.analysis_cache:
        cached_shapes, cached_warnings = ctx.call.analysis_cache[cache_key]
        # Replay warnings with current call site's dual-location formatting
        for func_warn in cached_warnings:
            formatted = _format_dual_location_warning(func_warn, func_name, line)
            warnings.append(formatted)
        return list(cached_shapes)  # Return copy

    # Mark function as currently being analyzed
    ctx.call.analyzing_functions.add(func_name)

    try:
        with ctx.snapshot_scope():
            # Analyze function body with fresh workspace
            func_env = Env()
            func_warnings: List['Diagnostic'] = []

            # Bind parameters to argument shapes + set up dimension aliases
            for param_name, arg, arg_shape in zip(sig.params, args, arg_shapes):
                func_env.set(param_name, arg_shape)

                # Dimension aliasing: extract dimension from arg expression (Const, Var, etc.)
                if isinstance(arg, IndexExpr):
                    caller_dim = expr_to_dim_ir(arg.expr, env)
                    if caller_dim is not None:
                        func_env.dim_aliases[param_name] = caller_dim

            # Pre-scan body for nested FunctionDef nodes (enables forward references)
            from ir import FunctionDef as FunctionDefIR
            for stmt in sig.body:
                if isinstance(stmt, FunctionDefIR):
                    nested_sig = FunctionSignature(
                        name=stmt.name,
                        params=stmt.params,
                        output_vars=stmt.output_vars,
                        body=stmt.body,
                    )
                    ctx.call.nested_function_registry[stmt.name] = nested_sig

            # Analyze function body (inherit fixpoint setting)
            try:
                for stmt in sig.body:
                    analyze_stmt_ir(stmt, func_env, func_warnings, ctx)
            except EarlyReturn:
                pass  # Function returned early — outputs are current env values

            # Extract return values from output variables
            result_shapes = []
            for out_var in sig.output_vars:
                shape = func_env.get(out_var)
                # Convert bottom (unset output var) to unknown
                if shape.is_bottom():
                    result_shapes.append(Shape.unknown())
                else:
                    result_shapes.append(shape)

            # Return output shapes (or single unknown if no outputs)
            result = result_shapes if result_shapes else [Shape.unknown()]

            # Store in cache before formatting warnings
            ctx.call.analysis_cache[cache_key] = (result, list(func_warnings))

            # Format warnings for current call site
            for func_warn in func_warnings:
                formatted = _format_dual_location_warning(func_warn, func_name, line)
                warnings.append(formatted)

            return result
    finally:
        # Remove function from analyzing set
        ctx.call.analyzing_functions.discard(func_name)


def analyze_nested_function_call(
    func_name: str,
    args: List[IndexArg],
    line: int,
    parent_env: Env,
    warnings: List['Diagnostic'],
    ctx: AnalysisContext
) -> List[Shape]:
    """Analyze a nested function call with read access to the parent workspace.

    Mirrors analyze_function_call but uses parent_env.push_scope() so that the
    nested function body can read variables from the parent function's workspace.
    Parameters are bound in the child scope, shadowing parent vars of the same name.
    Write-back to the parent workspace is deferred to Phase 2.

    Args:
        func_name: Name of the nested function to call
        args: Argument list from Apply node
        line: Call site line number
        parent_env: Parent function's environment (nested function reads from it)
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        List of output shapes (one per output_var in nested function signature)
    """
    from analysis.eval_expr import eval_expr_ir, _eval_index_arg_to_shape
    from analysis.stmt_analysis import analyze_stmt_ir

    if func_name not in ctx.call.nested_function_registry:
        return [Shape.unknown()]

    sig = ctx.call.nested_function_registry[func_name]

    # Check argument count
    if len(args) != len(sig.params):
        warnings.append(diag.warn_function_arg_count_mismatch(
            line, func_name, expected=len(sig.params), got=len(args)
        ))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Recursion guard
    if func_name in ctx.call.analyzing_functions:
        warnings.append(diag.warn_recursive_function(line, func_name))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Evaluate arg shapes for cache key
    arg_shapes = tuple(_eval_index_arg_to_shape(arg, parent_env, warnings, ctx) for arg in args)
    arg_dim_aliases = tuple(
        (param, expr_to_dim_ir(arg.expr, parent_env)) if isinstance(arg, IndexExpr) else (param, None)
        for param, arg in zip(sig.params, args)
    )
    cache_key = ("nested", func_name, arg_shapes, arg_dim_aliases)

    # Check cache
    if cache_key in ctx.call.analysis_cache:
        cached_shapes, cached_warnings = ctx.call.analysis_cache[cache_key]
        for func_warn in cached_warnings:
            formatted = _format_dual_location_warning(func_warn, func_name, line)
            warnings.append(formatted)
        return list(cached_shapes)

    ctx.call.analyzing_functions.add(func_name)

    try:
        with ctx.snapshot_scope():
            # Create child scope with parent env — gives read access to parent workspace
            func_env = parent_env.push_scope()
            func_warnings: List['Diagnostic'] = []

            # Bind parameters in child scope (shadow parent vars of same name)
            for param_name, arg, arg_shape in zip(sig.params, args, arg_shapes):
                func_env.set(param_name, arg_shape)

                if isinstance(arg, IndexExpr):
                    caller_dim = expr_to_dim_ir(arg.expr, parent_env)
                    if caller_dim is not None:
                        func_env.dim_aliases[param_name] = caller_dim

            # Pre-scan body for nested-nested FunctionDef nodes (forward references)
            from ir import FunctionDef as FunctionDefIR
            for stmt in sig.body:
                if isinstance(stmt, FunctionDefIR):
                    nested_sig = FunctionSignature(
                        name=stmt.name,
                        params=stmt.params,
                        output_vars=stmt.output_vars,
                        body=stmt.body,
                    )
                    ctx.call.nested_function_registry[stmt.name] = nested_sig

            # Analyze function body
            try:
                for stmt in sig.body:
                    analyze_stmt_ir(stmt, func_env, func_warnings, ctx)
            except EarlyReturn:
                pass

            # Write-back: flush modified parent-visible variables to parent env.
            # Skip parameters (they shadow parent vars intentionally).
            param_set = set(sig.params)
            for var_name, shape in func_env.bindings.items():
                if var_name not in param_set and var_name in parent_env:
                    parent_env.set(var_name, shape)

            # Extract return values from output variables
            result_shapes = []
            for out_var in sig.output_vars:
                shape = func_env.get(out_var)
                if shape.is_bottom():
                    result_shapes.append(Shape.unknown())
                else:
                    result_shapes.append(shape)

            result = result_shapes if result_shapes else [Shape.unknown()]

            ctx.call.analysis_cache[cache_key] = (result, list(func_warnings))

            for func_warn in func_warnings:
                formatted = _format_dual_location_warning(func_warn, func_name, line)
                warnings.append(formatted)

            return result
    finally:
        ctx.call.analyzing_functions.discard(func_name)


def analyze_external_function_call(
    fname: str,
    ext_sig: 'ExternalSignature',
    args: list,
    line: int,
    env: Env,
    warnings: List['Diagnostic'],
    ctx: AnalysisContext
) -> List[Shape]:
    """Analyze an external function call by parsing and analyzing the external file.

    Uses polymorphic caching, cross-file recursion guard, and registry swap
    for subfunction isolation. External body warnings are suppressed.

    Args:
        fname: Function name (filename stem)
        ext_sig: External signature from workspace scanning
        args: Argument list from Apply node
        line: Call site line number
        env: Caller's environment
        warnings: Caller's warning list (only call-site warnings appended)
        ctx: Analysis context

    Returns:
        List of output shapes
    """
    from analysis.eval_expr import _eval_index_arg_to_shape
    from analysis.stmt_analysis import analyze_stmt_ir
    from analysis.workspace import load_external_function

    # Cross-file recursion guard
    if fname in ctx.ws.analyzing_external:
        return [Shape.unknown()] * max(ext_sig.return_count, 1)

    # Load and parse external file
    loaded = load_external_function(ext_sig)
    if loaded is None:
        warnings.append(diag.warn_external_parse_error(line, fname, ext_sig.source_path))
        return [Shape.unknown()] * max(ext_sig.return_count, 1)

    primary_sig, subfunctions = loaded

    # Arg count check (caller-visible)
    if len(args) != len(primary_sig.params):
        warnings.append(diag.warn_function_arg_count_mismatch(
            line, fname, expected=len(primary_sig.params), got=len(args)
        ))
        return [Shape.unknown()] * max(len(primary_sig.output_vars), 1)

    # Evaluate arg shapes for cache key
    arg_shapes = tuple(_eval_index_arg_to_shape(arg, env, warnings, ctx) for arg in args)
    arg_dim_aliases = tuple(
        (param, expr_to_dim_ir(arg.expr, env)) if isinstance(arg, IndexExpr) else (param, None)
        for param, arg in zip(primary_sig.params, args)
    )
    cache_key = ("external", fname, arg_shapes, arg_dim_aliases)

    # Cache check (no warning replay — external warnings suppressed)
    if cache_key in ctx.call.analysis_cache:
        cached_shapes, _ = ctx.call.analysis_cache[cache_key]
        return list(cached_shapes)

    # Registry swap + recursion guard (site-specific; saved/restored outside snapshot_scope)
    saved_registry = ctx.call.function_registry
    ctx.call.function_registry = dict(subfunctions)
    ctx.ws.analyzing_external.add(fname)
    ctx.call.analyzing_functions.add(fname)  # So return statements work correctly inside external bodies

    try:
        with ctx.snapshot_scope():
            func_env = Env()
            func_warnings: List['Diagnostic'] = []  # Suppressed — not propagated to caller

            # Bind parameters with dimension aliasing
            for param_name, arg, arg_shape in zip(primary_sig.params, args, arg_shapes):
                func_env.set(param_name, arg_shape)
                if isinstance(arg, IndexExpr):
                    caller_dim = expr_to_dim_ir(arg.expr, env)
                    if caller_dim is not None:
                        func_env.dim_aliases[param_name] = caller_dim

            # Analyze function body
            try:
                for stmt in primary_sig.body:
                    analyze_stmt_ir(stmt, func_env, func_warnings, ctx)
            except EarlyReturn:
                pass

            # Extract return values
            result_shapes = []
            for out_var in primary_sig.output_vars:
                shape = func_env.get(out_var)
                result_shapes.append(Shape.unknown() if shape.is_bottom() else shape)

            result = result_shapes if result_shapes else [Shape.unknown()]
            ctx.call.analysis_cache[cache_key] = (result, [])
            return result
    finally:
        ctx.call.function_registry = saved_registry
        ctx.ws.analyzing_external.discard(fname)
        ctx.call.analyzing_functions.discard(fname)


def _analyze_loop_body(body: list, env: Env, warnings: List['Diagnostic'], ctx: AnalysisContext) -> None:
    """Analyze a loop body, optionally using widening-based fixed-point iteration.

    Modifies env in place. When ctx.fixpoint is True, uses a 3-phase widening algorithm:
    - Phase 1 (Discover): Analyze body once, widen conflicting dimensions
    - Phase 2 (Stabilize): Re-analyze with widened dims if widening changed anything
    - Phase 3 (Post-loop join): Model "loop may not execute" by widening pre-loop env with final env

    This guarantees convergence in <=2 iterations (vs unpredictable with iteration limit).
    Catches EarlyReturn/EarlyBreak/EarlyContinue at boundary (stops iteration, doesn't propagate).
    """
    from analysis.stmt_analysis import analyze_stmt_ir

    if not ctx.call.fixpoint:
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except (EarlyReturn, EarlyBreak, EarlyContinue):
            pass  # Stop iteration, don't propagate
        return

    # Phase 1 (Discover): Analyze body once to discover dimension conflicts
    pre_loop_env = env.copy()
    try:
        for s in body:
            analyze_stmt_ir(s, env, warnings, ctx)
    except (EarlyReturn, EarlyBreak, EarlyContinue):
        pass  # Phase 1 stopped early

    # Widen: stable dimensions preserved, conflicting dimensions -> None
    widened = widen_env(pre_loop_env, env)

    # Phase 2 (Stabilize): Re-analyze if widening changed anything
    # (widened dims like None x 1 should stabilize immediately in body)
    if not env.local_bindings_equal(widened):
        env.replace_local(widened)
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except (EarlyReturn, EarlyBreak, EarlyContinue):
            pass  # Phase 2 stopped early

    # Phase 3 (Post-loop join): Model "loop may execute 0 times"
    # Use widen_env (same operator) to join pre-loop and post-loop states
    final = widen_env(pre_loop_env, env)
    env.replace_local(final)
