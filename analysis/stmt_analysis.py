# Ethan Doughty
# stmt_analysis.py
"""Statement analysis — dispatch and environment updates for IR statements."""

from __future__ import annotations
from typing import List

from analysis.builtins import KNOWN_BUILTINS
from analysis.context import EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.dim_extract import _update_struct_field
from analysis.eval_expr import eval_expr_ir, _eval_index_arg_to_shape
from analysis.func_analysis import analyze_function_call, _analyze_loop_body
from analysis.constraints import snapshot_constraints, join_constraints

from ir import (
    Stmt,
    Assign, StructAssign, CellAssign, ExprStmt, While, For, If, IfChain, Switch, Try, Break, Continue,
    OpaqueStmt, FunctionDef, AssignMulti, Return,
    Apply, Var, Const, IndexExpr,
)

import analysis.diagnostics as diag
from runtime.env import Env, join_env
from runtime.shapes import Shape
from analysis.analysis_core import shapes_definitely_incompatible


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

        # Snapshot constraints before branching
        baseline_constraints = snapshot_constraints(ctx)

        # Analyze all branches, tracking which ones returned/broke/continued
        all_bodies = list(stmt.bodies) + [stmt.else_body]
        branch_envs = []
        branch_constraints = []
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
            branch_constraints.append(frozenset(ctx.constraints))
            returned_flags.append(returned)

        # If ALL branches returned/broke, propagate
        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        # Join only non-returned branches (environments and constraints)
        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        live_constraints = [c for c, r in zip(branch_constraints, returned_flags) if not r]

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

        return env

    if isinstance(stmt, Switch):
        _ = eval_expr_ir(stmt.expr, env, warnings, ctx)
        for case_val, _ in stmt.cases:
            _ = eval_expr_ir(case_val, env, warnings, ctx)

        # Snapshot constraints before branching
        baseline_constraints = snapshot_constraints(ctx)

        all_bodies = [body for _, body in stmt.cases] + [stmt.otherwise]
        branch_envs = []
        branch_constraints = []
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
            branch_constraints.append(frozenset(ctx.constraints))
            returned_flags.append(returned)

        if all(returned_flags):
            if deferred_exception:
                raise type(deferred_exception)()
            raise EarlyReturn()

        # Join only non-returned branches (environments and constraints)
        live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
        live_constraints = [c for c, r in zip(branch_constraints, returned_flags) if not r]

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
