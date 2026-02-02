# frontend/pipeline.py
"""Convenience functions for the Mini-MATLAB analysis pipeline."""

from __future__ import annotations
from typing import Any

from frontend.matlab_parser import parse_matlab
from frontend.lower_ir import lower_program
from ir import Program as IRProgram


def parse_syntax(src: str) -> Any:
    """Parse Mini-MATLAB source code to syntax AST.

    Args:
        src: Source code string

    Returns:
        List-based syntax AST
    """
    return parse_matlab(src)


def lower_to_ir(ast: Any) -> IRProgram:
    """Lower syntax AST to typed IR AST.

    Args:
        ast: List-based syntax AST

    Returns:
        Typed IR Program
    """
    return lower_program(ast)