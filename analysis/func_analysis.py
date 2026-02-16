# Ethan Doughty
# func_analysis.py
"""Function call analysis, loop body analysis, and warning formatting."""

from __future__ import annotations
from typing import List

from ir import IndexArg, IndexExpr

import analysis.diagnostics as diag
from analysis.context import FunctionSignature, EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.dim_extract import expr_to_dim_ir
from analysis.constraints import snapshot_constraints
from runtime.env import Env, widen_env
from runtime.shapes import Shape


def _format_dual_location_warning(func_warn: str, func_name: str, call_line: int) -> str:
    """Reformat a function-internal warning with call-site context.

    Args:
        func_warn: Warning message from function body
        func_name: Name of the called function
        call_line: Line number of the call site

    Returns:
        Warning message with dual-location context
    """
    # Extract warning line from func_warn
    # Formats: "W_... line N: ..." or "Line N: ..."
    # Check for both " line " (lowercase) and "Line " (capitalized at start)
    body_line = None
    prefix = None
    message = None

    if " line " in func_warn:
        # Format: "W_... line N: message"
        parts = func_warn.split(" line ", 1)
        prefix = parts[0]
        rest = parts[1]
        if ": " in rest:
            body_line = rest.split(": ", 1)[0].split(" ")[0]
            message = rest.split(": ", 1)[1]
    elif func_warn.startswith("Line "):
        # Format: "Line N: message"
        rest = func_warn[5:]  # Skip "Line "
        if ": " in rest:
            body_line = rest.split(": ", 1)[0]
            message = rest.split(": ", 1)[1]
            prefix = "Line"

    if body_line and message and prefix:
        # Check if already has call context
        if "(in " not in func_warn:
            # Add call context
            if prefix == "Line":
                return f"Line {body_line} (in {func_name}, called from line {call_line}): {message}"
            else:
                return f"{prefix} line {body_line} (in {func_name}, called from line {call_line}): {message}"
        else:
            # Already has call context, return as-is
            return func_warn
    else:
        # Could not parse, return as-is
        return func_warn


def analyze_function_call(
    func_name: str,
    args: List[IndexArg],
    line: int,
    env: Env,
    warnings: List[str],
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

    if func_name not in ctx.function_registry:
        # Should not reach here (checked by caller)
        return [Shape.unknown()]

    sig = ctx.function_registry[func_name]

    # Check argument count
    if len(args) != len(sig.params):
        warnings.append(diag.warn_function_arg_count_mismatch(
            line, func_name, expected=len(sig.params), got=len(args)
        ))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Recursion guard
    if func_name in ctx.analyzing_functions:
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
    if cache_key in ctx.analysis_cache:
        cached_shapes, cached_warnings = ctx.analysis_cache[cache_key]
        # Replay warnings with current call site's dual-location formatting
        for func_warn in cached_warnings:
            formatted = _format_dual_location_warning(func_warn, func_name, line)
            warnings.append(formatted)
        return list(cached_shapes)  # Return copy

    # Mark function as currently being analyzed
    ctx.analyzing_functions.add(func_name)

    # Snapshot constraints before analyzing function body (for isolation)
    baseline_constraints = snapshot_constraints(ctx)
    baseline_provenance = dict(ctx.constraint_provenance)

    try:
        # Analyze function body with fresh workspace
        func_env = Env()
        func_warnings: List[str] = []

        # Bind parameters to argument shapes + set up dimension aliases
        for param_name, arg, arg_shape in zip(sig.params, args, arg_shapes):
            func_env.set(param_name, arg_shape)

            # Dimension aliasing: extract dimension from arg expression (Const, Var, etc.)
            if isinstance(arg, IndexExpr):
                caller_dim = expr_to_dim_ir(arg.expr, env)
                if caller_dim is not None:
                    func_env.dim_aliases[param_name] = caller_dim

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
        ctx.analysis_cache[cache_key] = (result, list(func_warnings))

        # Format warnings for current call site
        for func_warn in func_warnings:
            formatted = _format_dual_location_warning(func_warn, func_name, line)
            warnings.append(formatted)

        return result

    finally:
        # Restore constraints (discard function-internal constraints)
        ctx.constraints = set(baseline_constraints)
        ctx.constraint_provenance = baseline_provenance

        # Remove function from analyzing set
        ctx.analyzing_functions.discard(func_name)


def _analyze_loop_body(body: list, env: Env, warnings: List[str], ctx: AnalysisContext) -> None:
    """Analyze a loop body, optionally using widening-based fixed-point iteration.

    Modifies env in place. When ctx.fixpoint is True, uses a 3-phase widening algorithm:
    - Phase 1 (Discover): Analyze body once, widen conflicting dimensions
    - Phase 2 (Stabilize): Re-analyze with widened dims if widening changed anything
    - Phase 3 (Post-loop join): Model "loop may not execute" by widening pre-loop env with final env

    This guarantees convergence in <=2 iterations (vs unpredictable with iteration limit).
    Catches EarlyReturn/EarlyBreak/EarlyContinue at boundary (stops iteration, doesn't propagate).
    """
    from analysis.stmt_analysis import analyze_stmt_ir

    if not ctx.fixpoint:
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
    if widened.bindings != env.bindings:
        env.bindings = widened.bindings
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except (EarlyReturn, EarlyBreak, EarlyContinue):
            pass  # Phase 2 stopped early

    # Phase 3 (Post-loop join): Model "loop may execute 0 times"
    # Use widen_env (same operator) to join pre-loop and post-loop states
    final = widen_env(pre_loop_env, env)
    env.bindings = final.bindings
