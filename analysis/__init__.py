# Ethan Doughty
# analysis/__init__.py
"""Analysis package — static shape inference for MATLAB."""

from __future__ import annotations
from typing import List, Tuple

from legacy.analysis_legacy import analyze_program as analyze_program_legacy

from analysis.context import FunctionSignature, EarlyReturn, EarlyBreak, EarlyContinue, AnalysisContext
from analysis.stmt_analysis import analyze_stmt_ir
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


# Default to legacy analyzer for backwards compatibility
analyze_program = analyze_program_legacy

__all__ = ["analyze_program", "analyze_program_ir", "analyze_program_legacy"]
