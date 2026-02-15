# Ethan Doughty
# analysis_ir.py
"""IR-based static shape analyzer for Mini-MATLAB.

This module performs shape inference on the typed IR AST, tracking matrix
dimensions and detecting dimension mismatches at compile time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set

from analysis.builtins import KNOWN_BUILTINS


from ir import (
    Program, Stmt,
    Assign, StructAssign, ExprStmt, While, For, If, IfChain, Switch, Try, Break, Continue,
    OpaqueStmt, FunctionDef, AssignMulti, Return,
    Expr, Var, Const, StringLit, FieldAccess, Lambda, FuncHandle, MatrixLit, Apply, Transpose, Neg, BinOp,
    IndexArg, Colon, Range, IndexExpr,
)

import analysis.diagnostics as diag
from runtime.env import Env, join_env, widen_env
from runtime.shapes import Shape, Dim, shape_of_zeros, shape_of_ones, join_dim, mul_dim, add_dim
from analysis.analysis_core import shapes_definitely_incompatible
from analysis.matrix_literals import infer_matrix_literal_shape, as_matrix_shape, dims_definitely_conflict


@dataclass
class FunctionSignature:
    """Registered user-defined function signature."""
    name: str
    params: List[str]
    output_vars: List[str]
    body: List[Stmt]

class EarlyReturn(Exception):
    """Raised by return statement to exit function body analysis."""
    pass

class EarlyBreak(Exception):
    """Raised by break statement to exit loop."""
    pass

class EarlyContinue(Exception):
    """Raised by continue statement to skip to next iteration."""
    pass

@dataclass
class AnalysisContext:
    """Threaded analysis state (replaces loose parameters)."""
    function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)
    analyzing_functions: Set[str] = field(default_factory=set)
    analysis_cache: Dict[tuple, tuple] = field(default_factory=dict)
    fixpoint: bool = False
    _lambda_closures: Dict[int, Env] = field(default_factory=dict)
    _next_lambda_id: int = 0


def analyze_program_ir(program: Program, fixpoint: bool = False, ctx: AnalysisContext = None) -> Tuple[Env, List[str]]:
    """Analyze a complete Mini-MATLAB program for shape consistency.

    Two-pass analysis:
    1. Register all function definitions
    2. Analyze script statements (non-function statements in program body)

    Args:
        program: IR program to analyze
        fixpoint: If True, use fixed-point iteration for loop analysis
        ctx: Analysis context (created if not provided)

    Returns:
        Tuple of (final environment, list of warning messages)
    """
    if ctx is None:
        ctx = AnalysisContext(fixpoint=fixpoint)

    env = Env()
    warnings: List[str] = []

    # Pass 1: Register function definitions
    for item in program.body:
        if isinstance(item, FunctionDef):
            ctx.function_registry[item.name] = FunctionSignature(
                name=item.name,
                params=item.params,
                output_vars=item.output_vars,
                body=item.body
            )

    # Pass 2: Analyze script statements (non-functions)
    try:
        for item in program.body:
            if not isinstance(item, FunctionDef):
                analyze_stmt_ir(item, env, warnings, ctx)
    except EarlyReturn:
        pass  # Script-level return stops analysis
    except (EarlyBreak, EarlyContinue):
        pass  # Break/continue outside loop (graceful degradation)

    # Deduplicate warnings while preserving order
    warnings = list(dict.fromkeys(warnings))
    return env, warnings


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
    cache_key = (func_name, arg_shapes)

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

    try:
        # Analyze function body with fresh workspace
        func_env = Env()
        func_warnings: List[str] = []

        # Bind parameters to argument shapes + set up dimension aliases
        for param_name, arg, arg_shape in zip(sig.params, args, arg_shapes):
            func_env.set(param_name, arg_shape)

            # Dimension aliasing: if arg is a Var, extract its dimension
            if isinstance(arg, IndexExpr) and isinstance(arg.expr, Var):
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


def analyze_stmt_ir(stmt: Stmt, env: Env, warnings: List[str], ctx: AnalysisContext) -> Env:
    """Analyze a statement and update environment with inferred shapes.

    Args:
        stmt: Statement to analyze
        env: Current environment (modified in place)
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        Updated environment
    """
    if isinstance(stmt, Assign):
        new_shape = eval_expr_ir(stmt.expr, env, warnings, ctx)
        old_shape = env.get(stmt.name)

        if stmt.name in env.bindings and shapes_definitely_incompatible(old_shape, new_shape):
            warnings.append(diag.warn_reassign_incompatible(stmt.line, stmt.name, new_shape, old_shape))

        env.set(stmt.name, new_shape)
        return env

    if isinstance(stmt, StructAssign):
        # Struct field assignment: s.field = expr or s.a.b = expr
        # Evaluate RHS
        rhs_shape = eval_expr_ir(stmt.expr, env, warnings, ctx)

        # Get current base variable shape
        base_shape = env.get(stmt.base_name)

        # Walk the field chain to update nested struct
        updated_shape = _update_struct_field(base_shape, stmt.fields, rhs_shape, stmt.line, warnings)

        env.set(stmt.base_name, updated_shape)
        return env

    if isinstance(stmt, ExprStmt):
        _ = eval_expr_ir(stmt.expr, env, warnings, ctx)
        return env

    if isinstance(stmt, While):
        _ = eval_expr_ir(stmt.cond, env, warnings, ctx)
        _analyze_loop_body(stmt.body, env, warnings, ctx)
        return env

    if isinstance(stmt, For):
        # Bind loop variable to scalar
        env.set(stmt.var, Shape.scalar())
        # Evaluate iterator expression for side effects
        _ = eval_expr_ir(stmt.it, env, warnings, ctx)
        _analyze_loop_body(stmt.body, env, warnings, ctx)
        return env

    if isinstance(stmt, If):
        _ = eval_expr_ir(stmt.cond, env, warnings, ctx)

        then_env = env.copy()
        else_env = env.copy()

        then_returned = False
        else_returned = False

        try:
            for s in stmt.then_body:
                analyze_stmt_ir(s, then_env, warnings, ctx)
        except EarlyReturn:
            then_returned = True

        try:
            for s in stmt.else_body:
                analyze_stmt_ir(s, else_env, warnings, ctx)
        except EarlyReturn:
            else_returned = True

        if then_returned and else_returned:
            # Both branches return — propagate
            raise EarlyReturn()
        elif then_returned:
            # Only then returns — use else env
            env.bindings = else_env.bindings
        elif else_returned:
            # Only else returns — use then env
            env.bindings = then_env.bindings
        else:
            # Neither returns — normal join
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

    if isinstance(stmt, Return):
        if not ctx.analyzing_functions:
            # Script context: warn and stop
            warnings.append(diag.warn_return_outside_function(stmt.line))
        raise EarlyReturn()

    if isinstance(stmt, Break):
        raise EarlyBreak()

    if isinstance(stmt, Continue):
        raise EarlyContinue()

    if isinstance(stmt, IfChain):
        # Evaluate all conditions for side effects
        for cond in stmt.conditions:
            _ = eval_expr_ir(cond, env, warnings, ctx)

        # Analyze all branches, tracking which ones returned/broke/continued
        all_bodies = list(stmt.bodies) + [stmt.else_body]
        branch_envs = []
        returned_flags = []
        deferred_exception = None  # EarlyBreak or EarlyContinue to re-raise

        for body in all_bodies:
            branch_env = env.copy()
            returned = False
            try:
                for s in body:
                    analyze_stmt_ir(s, branch_env, warnings, ctx)
            except EarlyReturn:
                returned = True
            except (EarlyBreak, EarlyContinue) as exc:
                # Break/continue inside if inside loop: record and re-raise after join
                returned = True  # Exclude from join (same as EarlyReturn)
                deferred_exception = exc
            branch_envs.append(branch_env)
            returned_flags.append(returned)

        # If ALL branches returned/broke, propagate
        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        # Join only non-returned branches
        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        if live_envs:
            result = live_envs[0]
            for other in live_envs[1:]:
                result = join_env(result, other)
            env.bindings = result.bindings
        return env

    if isinstance(stmt, Switch):
        _ = eval_expr_ir(stmt.expr, env, warnings, ctx)
        for case_val, _ in stmt.cases:
            _ = eval_expr_ir(case_val, env, warnings, ctx)

        all_bodies = [body for _, body in stmt.cases] + [stmt.otherwise]
        branch_envs = []
        returned_flags = []
        deferred_exception = None

        for body in all_bodies:
            branch_env = env.copy()
            returned = False
            try:
                for s in body:
                    analyze_stmt_ir(s, branch_env, warnings, ctx)
            except EarlyReturn:
                returned = True
            except (EarlyBreak, EarlyContinue) as exc:
                returned = True
                deferred_exception = exc
            branch_envs.append(branch_env)
            returned_flags.append(returned)

        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        if live_envs:
            result = live_envs[0]
            for other in live_envs[1:]:
                result = join_env(result, other)
            env.bindings = result.bindings
        return env

    if isinstance(stmt, Try):
        pre_try_env = env.copy()

        # Analyze try block
        try_env = env.copy()
        try_returned = False
        deferred_exception = None
        try:
            for s in stmt.try_body:
                analyze_stmt_ir(s, try_env, warnings, ctx)
        except EarlyReturn:
            try_returned = True
        except (EarlyBreak, EarlyContinue) as exc:
            try_returned = True
            deferred_exception = exc

        # Analyze catch block (starts from pre-try state)
        catch_env = pre_try_env.copy()
        catch_returned = False
        try:
            for s in stmt.catch_body:
                analyze_stmt_ir(s, catch_env, warnings, ctx)
        except EarlyReturn:
            catch_returned = True
        except (EarlyBreak, EarlyContinue) as exc:
            catch_returned = True
            if not deferred_exception:
                deferred_exception = exc

        # Propagation logic (same as If handler)
        if try_returned and catch_returned:
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()
        elif try_returned:
            env.bindings = catch_env.bindings
        elif catch_returned:
            env.bindings = try_env.bindings
        else:
            result = join_env(try_env, catch_env)
            env.bindings = result.bindings
        return env

    if isinstance(stmt, FunctionDef):
        # Phase A stub: skip function definitions (Phase C will register them)
        return env

    if isinstance(stmt, AssignMulti):
        # Destructuring assignment: [a, b] = expr
        if not isinstance(stmt.expr, Apply) or not isinstance(stmt.expr.base, Var):
            warnings.append(diag.warn_multi_assign_non_call(stmt.line))
            for target in stmt.targets:
                env.set(target, Shape.unknown())
            return env

        fname = stmt.expr.base.name

        # Check if builtin (builtins don't support multiple returns)
        if fname in KNOWN_BUILTINS:
            warnings.append(diag.warn_multi_assign_builtin(stmt.line, fname))
            for target in stmt.targets:
                env.set(target, Shape.unknown())
            return env

        # Check function registry
        if fname in ctx.function_registry:
            output_shapes = analyze_function_call(fname, stmt.expr.args, stmt.line, env, warnings, ctx)

            if len(stmt.targets) != len(output_shapes):
                warnings.append(diag.warn_multi_assign_count_mismatch(
                    stmt.line, fname, expected=len(output_shapes), got=len(stmt.targets)
                ))
                for target in stmt.targets:
                    env.set(target, Shape.unknown())
            else:
                for target, shape in zip(stmt.targets, output_shapes):
                    env.set(target, shape)
            return env

        # Not a known function
        warnings.append(diag.warn_unknown_function(stmt.line, fname))
        for target in stmt.targets:
            env.set(target, Shape.unknown())
        return env

    return env


def _eval_index_arg_to_shape(arg: IndexArg, env: Env, warnings: List[str], ctx: AnalysisContext) -> Shape:
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

    Returns:
        Shape of the argument
    """
    if isinstance(arg, IndexExpr):
        return eval_expr_ir(arg.expr, env, warnings, ctx)
    elif isinstance(arg, Range):
        # Range creates a row vector; shape depends on the bounds
        # For simplicity, we assume ranges create 1 x N matrices
        # (The actual size depends on start/end evaluation)
        return Shape.matrix(1, None)  # 1 x unknown
    elif isinstance(arg, Colon):
        # Standalone colon cannot be evaluated
        return Shape.unknown()
    return Shape.unknown()


def eval_expr_ir(expr: Expr, env: Env, warnings: List[str], ctx: AnalysisContext) -> Shape:
    """Evaluate an expression to infer its shape.

    Args:
        expr: Expression to evaluate
        env: Current environment with variable shapes
        warnings: List to append warnings to
        ctx: Analysis context

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

    if isinstance(expr, MatrixLit):
        # Evaluate element shapes first (before conversion)
        raw_shape_rows = [
            [eval_expr_ir(e, env, warnings, ctx) for e in row]
            for row in expr.rows
        ]
        return infer_matrix_literal_shape(raw_shape_rows, expr.line, warnings)

    if isinstance(expr, Apply):
        line = expr.line

        # Check for indexing indicators: colon or range in args
        has_colon_or_range = any(isinstance(arg, (Colon, Range)) for arg in expr.args)

        # Check if base variable is a function handle (shadows builtins)
        if isinstance(expr.base, Var):
            base_var_name = expr.base.name
            base_var_shape = env.get(base_var_name)
            if base_var_shape.is_function_handle():
                # Function handle call: v0.12.0 limitation - return unknown
                warnings.append(diag.warn_lambda_call_approximate(line, base_var_name))
                return Shape.unknown()

        # Check if base is a known builtin function
        if isinstance(expr.base, Var):
            fname = expr.base.name
            if fname in KNOWN_BUILTINS:
                # Known builtin: try function call first (even with colon/range)
                # Ranges/colons in builtin args are valid (e.g., length(1:10))
                # Treat as a function call
                if fname in {"zeros", "ones"} and len(expr.args) == 1:
                    # zeros(n) / ones(n) → matrix[n x n]
                    try:
                        d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                        return shape_of_zeros(d, d) if fname == "zeros" else shape_of_ones(d, d)
                    except ValueError:
                        # Colon/Range in arg: treat as indexing
                        pass
                if fname in {"zeros", "ones"} and len(expr.args) == 2:
                    try:
                        r_dim = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                        c_dim = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
                        return shape_of_zeros(r_dim, c_dim) if fname == "zeros" else shape_of_ones(r_dim, c_dim)
                    except ValueError:
                        # Colon/Range in builtin args: treat as indexing, not call
                        pass
                if fname == "size":
                    if len(expr.args) == 1:
                        try:
                            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                            return Shape.matrix(1, 2)
                        except ValueError:
                            # Colon in arg: treat as indexing
                            pass
                    elif len(expr.args) == 2:
                        try:
                            eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                            return Shape.scalar()
                        except ValueError:
                            # Colon in arg: treat as indexing
                            pass
                if fname == "isscalar" and len(expr.args) == 1:
                    try:
                        eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                        return Shape.scalar()
                    except ValueError:
                        # Colon in arg: treat as indexing
                        pass
                # Matrix constructors: eye, rand, randn
                if fname in {"eye", "rand", "randn"} and len(expr.args) <= 2:
                    if len(expr.args) == 0:
                        return Shape.scalar()
                    elif len(expr.args) == 1:
                        try:
                            d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                            return Shape.matrix(d, d)  # n×n square matrix
                        except ValueError:
                            pass
                    elif len(expr.args) == 2:
                        try:
                            r = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
                            c = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
                            return Shape.matrix(r, c)
                        except ValueError:
                            pass
                # Element-wise operations: abs, sqrt (output shape = input shape)
                if fname in {"abs", "sqrt"} and len(expr.args) == 1:
                    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    return arg_shape
                # Transpose function: swap row/col dimensions
                if fname == "transpose" and len(expr.args) == 1:
                    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    if arg_shape.is_matrix():
                        return Shape.matrix(arg_shape.cols, arg_shape.rows)
                    return arg_shape
                # Query functions: length, numel (return scalar)
                if fname in {"length", "numel"} and len(expr.args) == 1:
                    _ = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    return Shape.scalar()
                # Reshape function: return matrix[m x n] from args 2 and 3
                if fname == "reshape" and len(expr.args) == 3:
                    try:
                        # Evaluate first arg for side effects, discard its shape
                        _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                        # Extract dimensions from args 2 and 3
                        m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
                        n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)
                        return Shape.matrix(m, n)
                    except ValueError:
                        # Colon/Range in args: cannot reshape with colons
                        pass
                # Repmat function: replicate matrix
                if fname == "repmat" and len(expr.args) == 3:
                    try:
                        # Evaluate first arg to get its shape
                        a_shape = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                        # Extract replication factors from args 2 and 3
                        m = expr_to_dim_ir(unwrap_arg(expr.args[1]), env)
                        n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)

                        # If a is unknown, result is unknown
                        if a_shape.is_unknown():
                            return Shape.unknown()

                        # If a is scalar, treat as 1x1 matrix
                        if a_shape.is_scalar():
                            a_rows = 1
                            a_cols = 1
                        else:
                            a_rows = a_shape.rows
                            a_cols = a_shape.cols

                        # Result is [a_rows * m x a_cols * n]
                        result_rows = mul_dim(a_rows, m)
                        result_cols = mul_dim(a_cols, n)
                        return Shape.matrix(result_rows, result_cols)
                    except ValueError:
                        # Colon/Range in args: cannot repmat with colons
                        pass
                # Scalar-returning operations: det, norm
                if fname in {"det", "norm"} and len(expr.args) == 1:
                    # Evaluate arg for side effects, return scalar
                    _ = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    return Shape.scalar()
                # diag: shape-dependent dispatch
                if fname == "diag" and len(expr.args) == 1:
                    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    if arg_shape.is_scalar():
                        # scalar → 1x1 matrix
                        return Shape.matrix(1, 1)
                    if arg_shape.is_matrix():
                        r, c = arg_shape.rows, arg_shape.cols
                        # Check if it's provably a vector (one dim is concretely 1)
                        if r == 1:
                            # Row vector 1×n → diagonal n×n
                            return Shape.matrix(c, c)
                        if c == 1:
                            # Column vector n×1 → diagonal n×n
                            return Shape.matrix(r, r)
                        # Matrix (neither dim is 1, or unknown dims) → extract diagonal
                        # Result is min(m,n) x 1, but we can't compute min symbolically
                        return Shape.matrix(None, 1)
                    # Unknown shape → unknown result
                    return Shape.unknown()
                # inv: pass-through for square matrices
                if fname == "inv" and len(expr.args) == 1:
                    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings, ctx)
                    if arg_shape.is_matrix():
                        r, c = arg_shape.rows, arg_shape.cols
                        # Check if square (same dims)
                        if r == c:
                            # Pass through shape for square matrices
                            return Shape.matrix(r, c)
                        # Non-square or unknown dims → unknown
                        return Shape.unknown()
                    # Scalar or unknown → unknown
                    return Shape.unknown()
                # linspace: row vector generator
                if fname == "linspace":
                    if len(expr.args) == 2:
                        # linspace(a, b) → 1 x 100 (MATLAB default)
                        try:
                            _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                            _ = eval_expr_ir(unwrap_arg(expr.args[1]), env, warnings, ctx)
                            return Shape.matrix(1, 100)
                        except ValueError:
                            # Colon/Range in args
                            pass
                    elif len(expr.args) == 3:
                        # linspace(a, b, n) → 1 x n
                        try:
                            _ = eval_expr_ir(unwrap_arg(expr.args[0]), env, warnings, ctx)
                            _ = eval_expr_ir(unwrap_arg(expr.args[1]), env, warnings, ctx)
                            n = expr_to_dim_ir(unwrap_arg(expr.args[2]), env)
                            return Shape.matrix(1, n)
                        except ValueError:
                            # Colon/Range in args
                            pass
                # Known builtin without a matching shape rule: return unknown silently
                return Shape.unknown()

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
        # Store snapshot of current environment as closure
        ctx._lambda_closures[lambda_id] = env
        return Shape.function_handle()

    if isinstance(expr, FuncHandle):
        # Named function handle: check if function exists
        if expr.name in ctx.function_registry or expr.name in KNOWN_BUILTINS:
            return Shape.function_handle()
        else:
            # Unknown function
            warnings.append(diag.warn_unknown_function(expr.line, expr.name))
            return Shape.function_handle()  # Still return function_handle shape

    if isinstance(expr, BinOp):
        op = expr.op
        line = expr.line
        left_shape = eval_expr_ir(expr.left, env, warnings, ctx)
        right_shape = eval_expr_ir(expr.right, env, warnings, ctx)
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

            r_extent = index_arg_to_extent_ir(a1, env, warnings, line, ctx)
            c_extent = index_arg_to_extent_ir(a2, env, warnings, line, ctx)

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
    ctx: AnalysisContext
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

        start_shape = eval_expr_ir(start_expr, env, warnings, ctx)
        end_shape = eval_expr_ir(end_expr, env, warnings, ctx)

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
        s = eval_expr_ir(arg.expr, env, warnings, ctx)
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
        # Check for dimension alias first (propagates caller's dim name)
        if expr.name in env.dim_aliases:
            return env.dim_aliases[expr.name]
        return expr.name
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
            # Negate right operand for subtraction
            if isinstance(right_dim, int):
                negated_right = -right_dim
            else:
                # Symbolic dimension: wrap in negation
                negated_right = f"-{right_dim}"
            return add_dim(left_dim, negated_right)
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
    warnings: List[str]
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

    # String arithmetic: string + string = numeric row vector (MATLAB behavior)
    if op == "+" and left.is_string() and right.is_string():
        return Shape.matrix(1, None)  # char + char = numeric row vector, length unknown

    # String + non-string: warning + unknown
    if op in {"+", "-", "*", ".*", "/", "./"} and (left.is_string() or right.is_string()):
        if not (left.is_string() and right.is_string() and op == "+"):
            warnings.append(diag.warn_string_arithmetic(line, op, left, right))
            return Shape.unknown()

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