# Ethan Doughty
# end_helpers.py
"""Utilities for resolving the End keyword in indexing expressions."""

from __future__ import annotations
from typing import Optional

from ir import Expr, End, Const, BinOp


def _binop_contains_end(expr: Expr) -> bool:
    """Recursively check if End appears in a BinOp tree."""
    if isinstance(expr, End):
        return True
    if isinstance(expr, BinOp):
        return _binop_contains_end(expr.left) or _binop_contains_end(expr.right)
    return False


def _eval_end_arithmetic(expr: Expr, end_value: int) -> Optional[int]:
    """Evaluate a BinOp tree with End resolved to end_value.

    Returns None if can't resolve (e.g., contains variables).
    """
    if isinstance(expr, End):
        return end_value
    if isinstance(expr, Const):
        return int(expr.value)
    if isinstance(expr, BinOp):
        left = _eval_end_arithmetic(expr.left, end_value)
        right = _eval_end_arithmetic(expr.right, end_value)
        if left is None or right is None:
            return None
        if expr.op == '+':
            return left + right
        if expr.op == '-':
            return left - right
        if expr.op == '*':
            return left * right
        if expr.op == '/':
            return left // right if right != 0 else None
        # Unsupported operator
        return None
    return None  # Can't resolve (e.g., Var in expression)
