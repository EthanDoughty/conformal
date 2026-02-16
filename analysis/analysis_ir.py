# Ethan Doughty
# analysis_ir.py
"""IR-based static shape analyzer for MATLAB — entry point facade.

Delegates to submodules for expression evaluation, statement analysis,
function call analysis, and builtin dispatch.
"""

from __future__ import annotations
from typing import List, Tuple

from analysis.context import FunctionSignature, EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.stmt_analysis import analyze_stmt_ir
from analysis.eval_expr import eval_expr_ir, _eval_index_arg_to_shape
from analysis.func_analysis import analyze_function_call, _analyze_loop_body

from ir import Program, FunctionDef
from runtime.env import Env


def analyze_program_ir(program: Program, fixpoint: bool = False, ctx: AnalysisContext = None) -> Tuple[Env, List[str]]:
    """Analyze a complete MATLAB program for shape consistency.

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
