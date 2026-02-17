# Ethan Doughty
# stmt_analysis.py
"""Statement analysis — dispatch and environment updates for IR statements."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from analysis.builtins import KNOWN_BUILTINS
from analysis.context import EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.dim_extract import _update_struct_field, extract_iteration_count
from analysis.eval_expr import eval_expr_ir, _eval_index_arg_to_shape
from analysis.func_analysis import analyze_function_call, _analyze_loop_body
from analysis.constraints import snapshot_constraints, join_constraints, validate_binding, try_extract_const_value

from ir import (
    Stmt, Expr,
    Assign, StructAssign, CellAssign, ExprStmt, While, For, If, IfChain, Switch, Try, Break, Continue,
    OpaqueStmt, FunctionDef, AssignMulti, Return,
    Apply, Var, Const, IndexExpr, MatrixLit, BinOp, Neg, Transpose, FieldAccess, Lambda, FuncHandle,
    End, CellLit, CurlyApply, StringLit,
)

import analysis.diagnostics as diag
from runtime.env import Env, join_env
from runtime.shapes import Shape, Dim, add_dim, mul_dim
from analysis.analysis_core import shapes_definitely_incompatible

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


@dataclass
class _AccumPattern:
    """Detected accumulation pattern in loop body."""
    var_name: str           # Accumulated variable name
    axis: str               # "vert" or "horz"
    delta_exprs: List[List[Expr]]  # Non-X rows/elements of MatrixLit
    line: int               # MatrixLit line (for delta evaluation)
    loop_var: str           # Loop variable (to check for self-reference in delta)


def _expr_mentions_var(expr: Expr, var_name: str) -> bool:
    """Check if an expression mentions a specific variable.

    Args:
        expr: Expression to check
        var_name: Variable name to search for

    Returns:
        True if var_name appears in expr
    """
    if isinstance(expr, Var):
        return expr.name == var_name
    if isinstance(expr, (Const, StringLit, End)):
        return False
    if isinstance(expr, Neg):
        return _expr_mentions_var(expr.operand, var_name)
    if isinstance(expr, Transpose):
        return _expr_mentions_var(expr.operand, var_name)
    if isinstance(expr, BinOp):
        return _expr_mentions_var(expr.left, var_name) or _expr_mentions_var(expr.right, var_name)
    if isinstance(expr, FieldAccess):
        return _expr_mentions_var(expr.base, var_name)
    if isinstance(expr, (Lambda, FuncHandle)):
        return False  # Lambdas/handles don't directly reference loop vars
    if isinstance(expr, MatrixLit):
        for row in expr.rows:
            for elem in row:
                if _expr_mentions_var(elem, var_name):
                    return True
        return False
    if isinstance(expr, CellLit):
        for row in expr.rows:
            for elem in row:
                if _expr_mentions_var(elem, var_name):
                    return True
        return False
    if isinstance(expr, (Apply, CurlyApply)):
        if _expr_mentions_var(expr.base, var_name):
            return True
        # Check args (simplified: assume IndexExpr wrapping)
        for arg in expr.args:
            if isinstance(arg, IndexExpr) and _expr_mentions_var(arg.expr, var_name):
                return True
        return False
    return False


def _detect_accumulation(loop_var: str, body: List[Stmt]) -> 'List[_AccumPattern]':
    """Detect accumulation patterns in loop body.

    Structural pattern match on top-level body statements only.
    Returns list of all valid patterns found.

    Args:
        loop_var: Loop variable name (to check for self-reference in delta)
        body: Loop body statements

    Returns:
        List of _AccumPattern instances (empty if none detected)
    """
    # Scan top-level assignments
    candidates = {}
    for stmt in body:
        if not isinstance(stmt, Assign):
            continue
        if not isinstance(stmt.expr, MatrixLit):
            continue
        var_name = stmt.name
        lit = stmt.expr

        # Count occurrences of Var(name=var_name) in literal
        count = 0
        for row in lit.rows:
            for elem in row:
                if isinstance(elem, Var) and elem.name == var_name:
                    count += 1

        if count != 1:
            continue  # Must appear exactly once

        # Check if X appears multiple times in body (multiple assignments)
        if var_name in candidates:
            # Multiple assignments to same var: disqualify
            candidates[var_name] = None
            continue

        # Detect vertcat or horzcat pattern
        if len(lit.rows) >= 2:
            # Vertcat candidate: first row must be [Var(X)]
            first_row = lit.rows[0]
            if len(first_row) == 1 and isinstance(first_row[0], Var) and first_row[0].name == var_name:
                # Valid vertcat pattern
                delta_exprs = lit.rows[1:]
                candidates[var_name] = _AccumPattern(var_name, "vert", delta_exprs, stmt.line, loop_var)
        elif len(lit.rows) == 1 and len(lit.rows[0]) >= 2:
            # Horzcat candidate: first element must be Var(X)
            first_elem = lit.rows[0][0]
            if isinstance(first_elem, Var) and first_elem.name == var_name:
                # Valid horzcat pattern
                delta_exprs = [lit.rows[0][1:]]
                candidates[var_name] = _AccumPattern(var_name, "horz", delta_exprs, stmt.line, loop_var)

    # Return all valid patterns
    return [pattern for pattern in candidates.values() if pattern is not None]


def _refine_accumulation(
    accum: _AccumPattern,
    iter_count: Dim,
    pre_loop_env: Env,
    post_loop_env: Env,
    warnings: List['Diagnostic'],
    ctx: AnalysisContext
) -> None:
    """Refine accumulation variable shape using algebraic computation.

    Modifies post_loop_env in place.

    Args:
        accum: Detected accumulation pattern
        iter_count: Iteration count (int, SymDim, or None)
        pre_loop_env: Environment before loop
        post_loop_env: Environment after loop (modified in place)
        warnings: List to append warnings to (not used for delta eval)
        ctx: Analysis context
    """
    if iter_count is None:
        return  # Can't refine without iteration count

    # Get initial shape
    init_shape = pre_loop_env.get(accum.var_name)
    if init_shape.is_bottom() or init_shape.is_unknown() or not init_shape.is_matrix():
        return  # Can't refine

    # Get current (widened) shape
    current_shape = post_loop_env.get(accum.var_name)
    if not current_shape.is_matrix():
        return

    # Check for self-reference in delta expressions
    for row in accum.delta_exprs:
        for elem in row:
            if _expr_mentions_var(elem, accum.var_name):
                return  # Delta references accumulated var: bail
            if _expr_mentions_var(elem, accum.loop_var):
                return  # Delta references loop var: bail

    # Evaluate delta in pre_loop_env (throwaway warnings)
    delta_warnings = []
    from analysis.matrix_literals import infer_matrix_literal_shape
    delta_shape_rows = []
    for row in accum.delta_exprs:
        delta_row_shapes = []
        for elem in row:
            elem_shape = eval_expr_ir(elem, pre_loop_env, delta_warnings, ctx)
            delta_row_shapes.append(elem_shape)
        delta_shape_rows.append(delta_row_shapes)

    delta_shape = infer_matrix_literal_shape(delta_shape_rows, accum.line, delta_warnings, ctx, pre_loop_env)

    if not delta_shape.is_matrix():
        return  # Delta didn't evaluate to matrix

    # Compute refined shape
    if accum.axis == "vert":
        # Vertical accumulation: rows = init_rows + iter_count * delta_rows
        init_rows = init_shape.rows
        delta_rows = delta_shape.rows
        if delta_rows is None:
            return  # Can't compute
        # Compute total added rows: iter_count * delta_rows
        total_added_rows = mul_dim(iter_count, delta_rows)
        refined_rows = add_dim(init_rows, total_added_rows)
        refined_cols = init_shape.cols  # Cols should match (enforced by concat rules)
        refined_shape = Shape.matrix(refined_rows, refined_cols)
    else:  # horz
        # Horizontal accumulation: cols = init_cols + iter_count * delta_cols
        init_cols = init_shape.cols
        delta_cols = delta_shape.cols
        if delta_cols is None:
            return  # Can't compute
        # Compute total added cols: iter_count * delta_cols
        total_added_cols = mul_dim(iter_count, delta_cols)
        refined_cols = add_dim(init_cols, total_added_cols)
        refined_rows = init_shape.rows  # Rows should match
        refined_shape = Shape.matrix(refined_rows, refined_cols)

    # Update environment only where current shape has None (was widened)
    if accum.axis == "vert":
        # Replace None row with refined row
        if current_shape.rows is None:
            post_loop_env.set(accum.var_name, refined_shape)
    else:  # horz
        # Replace None col with refined col
        if current_shape.cols is None:
            post_loop_env.set(accum.var_name, refined_shape)


def analyze_stmt_ir(stmt: Stmt, env: Env, warnings: List['Diagnostic'], ctx: AnalysisContext) -> Env:
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
        old_shape = env.get(stmt.name)
        new_shape = eval_expr_ir(stmt.expr, env, warnings, ctx)

        if stmt.name in env.bindings and shapes_definitely_incompatible(old_shape, new_shape):
            warnings.append(diag.warn_reassign_incompatible(stmt.line, stmt.name, new_shape, old_shape))

        env.set(stmt.name, new_shape)

        # Phase 2: Constraint validation trigger
        # Check if this is first binding (was bottom) of a scalar to a concrete value
        if old_shape.is_bottom() and new_shape.is_scalar():
            concrete_value = try_extract_const_value(stmt.expr)
            if concrete_value is not None:
                # Record scalar binding for future constraint checks
                ctx.scalar_bindings[stmt.name] = concrete_value
                # Validate against existing constraints
                validate_binding(ctx, env, stmt.name, concrete_value, warnings, stmt.line)

        # Interval tracking: update value_ranges for scalar assignments
        if new_shape.is_scalar():
            from analysis.eval_expr import _get_expr_interval
            interval = _get_expr_interval(stmt.expr, env, ctx)
            if interval is not None:
                ctx.value_ranges[stmt.name] = interval
            elif stmt.name in ctx.value_ranges:
                # Remove stale interval if we can't compute a new one
                del ctx.value_ranges[stmt.name]
        else:
            # Non-scalar: remove from value_ranges
            if stmt.name in ctx.value_ranges:
                del ctx.value_ranges[stmt.name]

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

    if isinstance(stmt, CellAssign):
        # Cell element assignment: c{i} = expr
        # Evaluate RHS
        rhs_shape = eval_expr_ir(stmt.expr, env, warnings, ctx)

        # Get current base variable shape
        base_shape = env.get(stmt.base_name)

        # Verify base is cell (or bottom for unbound variables)
        if base_shape.is_bottom():
            # Unbound variable: create cell with unknown dimensions, no element tracking
            env.set(stmt.base_name, Shape.cell(None, None, elements=None))
            return env
        elif not base_shape.is_cell():
            # Evaluate index args for side effects
            for arg in stmt.args:
                _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)
            # Base is not a cell: warn and keep original shape
            warnings.append(diag.warn_cell_assign_non_cell(stmt.line, stmt.base_name, base_shape))
            return env

        # Base is cell: try to update element tracking
        # Check if we can extract literal index
        if len(stmt.args) == 1:
            arg = stmt.args[0]
            if isinstance(arg, IndexExpr) and isinstance(arg.expr, Const):
                # Literal 1D index: update element tracking
                linear_idx = int(arg.expr.value) - 1  # Convert to 0-based

                # Get current elements
                if base_shape._elements is None:
                    current_elems = {}
                else:
                    current_elems = dict(base_shape._elements)

                # Update element at index
                current_elems[linear_idx] = rhs_shape

                # Create updated cell shape
                updated_shape = Shape.cell(base_shape.rows, base_shape.cols, elements=current_elems)
                env.set(stmt.base_name, updated_shape)
                return env

        # Dynamic index or 2D literal: drop element tracking (set _elements = None)
        # Evaluate args for side effects
        for arg in stmt.args:
            _ = _eval_index_arg_to_shape(arg, env, warnings, ctx, container_shape=base_shape)

        # Preserve container dimensions, drop element tracking
        updated_shape = Shape.cell(base_shape.rows, base_shape.cols, elements=None)
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

        # Record loop variable interval if iterator is a:b range with concrete bounds
        if isinstance(stmt.it, BinOp) and stmt.it.op == ':':
            from analysis.intervals import Interval
            lo_val = try_extract_const_value(stmt.it.left)
            hi_val = try_extract_const_value(stmt.it.right)
            if lo_val is not None and hi_val is not None:
                ctx.value_ranges[stmt.var] = Interval(lo_val, hi_val)

        # Fixpoint-only: accumulation refinement
        if ctx.fixpoint:
            pre_loop_env = env.copy()
            iter_count = extract_iteration_count(stmt.it, env)
            accum_patterns = _detect_accumulation(stmt.var, stmt.body) if iter_count is not None else []
        else:
            accum_patterns = []

        _analyze_loop_body(stmt.body, env, warnings, ctx)

        # Fixpoint-only: refine accumulation variables
        if ctx.fixpoint:
            for accum in accum_patterns:
                _refine_accumulation(accum, iter_count, pre_loop_env, env, warnings, ctx)

        return env

    if isinstance(stmt, If):
        _ = eval_expr_ir(stmt.cond, env, warnings, ctx)

        # Snapshot constraints and value_ranges before branching
        baseline_constraints = snapshot_constraints(ctx)
        baseline_ranges = dict(ctx.value_ranges)

        then_env = env.copy()
        else_env = env.copy()

        then_returned = False
        else_returned = False

        try:
            for s in stmt.then_body:
                analyze_stmt_ir(s, then_env, warnings, ctx)
        except EarlyReturn:
            then_returned = True
        then_constraints = frozenset(ctx.constraints)
        then_ranges = dict(ctx.value_ranges)

        # Reset constraints and value_ranges to baseline for else branch
        ctx.constraints = set(baseline_constraints)
        ctx.value_ranges = dict(baseline_ranges)

        try:
            for s in stmt.else_body:
                analyze_stmt_ir(s, else_env, warnings, ctx)
        except EarlyReturn:
            else_returned = True
        else_constraints = frozenset(ctx.constraints)
        else_ranges = dict(ctx.value_ranges)

        if then_returned and else_returned:
            # Both branches return — propagate
            raise EarlyReturn()
        elif then_returned:
            # Only then returns — use else env, constraints, and value_ranges
            env.bindings = else_env.bindings
            ctx.constraints = set(else_constraints)
            ctx.value_ranges = else_ranges
        elif else_returned:
            # Only else returns — use then env, constraints, and value_ranges
            env.bindings = then_env.bindings
            ctx.constraints = set(then_constraints)
            ctx.value_ranges = then_ranges
        else:
            # Neither returns — normal join
            merged = join_env(then_env, else_env)
            env.bindings = merged.bindings

            # Join constraints
            joined_constraints = join_constraints(baseline_constraints, [then_constraints, else_constraints])
            ctx.constraints = joined_constraints

            # Update provenance
            new_provenance = {}
            for constraint in joined_constraints:
                if constraint in ctx.constraint_provenance:
                    new_provenance[constraint] = ctx.constraint_provenance[constraint]
            ctx.constraint_provenance = new_provenance

            # Join value_ranges
            from analysis.intervals import join_value_ranges
            ctx.value_ranges = join_value_ranges(baseline_ranges, [then_ranges, else_ranges])

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

        # Snapshot constraints and value_ranges before branching
        baseline_constraints = snapshot_constraints(ctx)
        baseline_ranges = dict(ctx.value_ranges)

        # Analyze all branches, tracking which ones returned/broke/continued
        all_bodies = list(stmt.bodies) + [stmt.else_body]
        branch_envs = []
        branch_constraints = []
        branch_ranges = []
        returned_flags = []
        deferred_exception = None  # EarlyBreak or EarlyContinue to re-raise

        for body in all_bodies:
            ctx.constraints = set(baseline_constraints)  # Reset to baseline (prevent cross-branch contamination)
            ctx.value_ranges = dict(baseline_ranges)
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
            branch_constraints.append(frozenset(ctx.constraints))
            branch_ranges.append(dict(ctx.value_ranges))
            returned_flags.append(returned)

        # If ALL branches returned/broke, propagate
        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        # Join only non-returned branches (environments, constraints, and value_ranges)
        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        live_constraints = [c for c, r in zip(branch_constraints, returned_flags) if not r]
        live_ranges = [vr for vr, r in zip(branch_ranges, returned_flags) if not r]

        if live_envs:
            # Join environments
            result = live_envs[0]
            for other in live_envs[1:]:
                result = join_env(result, other)
            env.bindings = result.bindings

            # Join constraints
            joined_constraints = join_constraints(baseline_constraints, live_constraints)
            ctx.constraints = joined_constraints

            # Update provenance: keep only provenance for constraints that survived
            new_provenance = {}
            for constraint in joined_constraints:
                if constraint in ctx.constraint_provenance:
                    new_provenance[constraint] = ctx.constraint_provenance[constraint]
            ctx.constraint_provenance = new_provenance

            # Join value_ranges
            from analysis.intervals import join_value_ranges
            ctx.value_ranges = join_value_ranges(baseline_ranges, live_ranges)

        return env

    if isinstance(stmt, Switch):
        _ = eval_expr_ir(stmt.expr, env, warnings, ctx)
        for case_val, _ in stmt.cases:
            _ = eval_expr_ir(case_val, env, warnings, ctx)

        # Snapshot constraints and value_ranges before branching
        baseline_constraints = snapshot_constraints(ctx)
        baseline_ranges = dict(ctx.value_ranges)

        all_bodies = [body for _, body in stmt.cases] + [stmt.otherwise]
        branch_envs = []
        branch_constraints = []
        branch_ranges = []
        returned_flags = []
        deferred_exception = None

        for body in all_bodies:
            ctx.constraints = set(baseline_constraints)  # Reset to baseline (prevent cross-branch contamination)
            ctx.value_ranges = dict(baseline_ranges)
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
            branch_constraints.append(frozenset(ctx.constraints))
            branch_ranges.append(dict(ctx.value_ranges))
            returned_flags.append(returned)

        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        # Join only non-returned branches (environments, constraints, and value_ranges)
        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        live_constraints = [c for c, r in zip(branch_constraints, returned_flags) if not r]
        live_ranges = [vr for vr, r in zip(branch_ranges, returned_flags) if not r]

        if live_envs:
            # Join environments
            result = live_envs[0]
            for other in live_envs[1:]:
                result = join_env(result, other)
            env.bindings = result.bindings

            # Join constraints
            joined_constraints = join_constraints(baseline_constraints, live_constraints)
            ctx.constraints = joined_constraints

            # Update provenance: keep only provenance for constraints that survived
            new_provenance = {}
            for constraint in joined_constraints:
                if constraint in ctx.constraint_provenance:
                    new_provenance[constraint] = ctx.constraint_provenance[constraint]
            ctx.constraint_provenance = new_provenance

            # Join value_ranges
            from analysis.intervals import join_value_ranges
            ctx.value_ranges = join_value_ranges(baseline_ranges, live_ranges)

        return env

    if isinstance(stmt, Try):
        # Snapshot constraints and value_ranges before branching
        baseline_constraints = snapshot_constraints(ctx)
        baseline_ranges = dict(ctx.value_ranges)
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
        try_constraints = frozenset(ctx.constraints)
        try_ranges = dict(ctx.value_ranges)

        # Reset to baseline for catch block
        ctx.constraints = set(baseline_constraints)
        ctx.value_ranges = dict(baseline_ranges)

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
        catch_constraints = frozenset(ctx.constraints)
        catch_ranges = dict(ctx.value_ranges)

        # Propagation logic (same as If handler)
        if try_returned and catch_returned:
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()
        elif try_returned:
            env.bindings = catch_env.bindings
            ctx.constraints = set(catch_constraints)
            ctx.value_ranges = catch_ranges
        elif catch_returned:
            env.bindings = try_env.bindings
            ctx.constraints = set(try_constraints)
            ctx.value_ranges = try_ranges
        else:
            result = join_env(try_env, catch_env)
            env.bindings = result.bindings

            # Join constraints
            joined_constraints = join_constraints(baseline_constraints, [try_constraints, catch_constraints])
            ctx.constraints = joined_constraints

            # Update provenance
            new_provenance = {}
            for constraint in joined_constraints:
                if constraint in ctx.constraint_provenance:
                    new_provenance[constraint] = ctx.constraint_provenance[constraint]
            ctx.constraint_provenance = new_provenance

            # Join value_ranges
            from analysis.intervals import join_value_ranges
            ctx.value_ranges = join_value_ranges(baseline_ranges, [try_ranges, catch_ranges])

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

        # Check external functions (workspace scanning — cross-file analysis)
        if fname in ctx.external_functions:
            from analysis.func_analysis import analyze_external_function_call
            output_shapes = analyze_external_function_call(
                fname, ctx.external_functions[fname], stmt.expr.args, stmt.line, env, warnings, ctx)
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
