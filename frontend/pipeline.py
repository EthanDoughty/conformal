# frontend/pipeline.py
"""Convenience functions for the MATLAB analysis pipeline."""

from __future__ import annotations

from frontend.matlab_parser import parse_matlab
from ir.ir import Program


def parse_syntax(src: str) -> Program:
    """Parse MATLAB source code to IR Program.

    Args:
        src: Source code string

    Returns:
        IR Program (parse_matlab now returns IR directly)
    """
    return parse_matlab(src)
