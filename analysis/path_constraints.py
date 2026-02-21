# Ethan Doughty
# path_constraints.py
"""Branch-aware path constraint stack for witness generation.

Tracks which branch conditions were active at each warning site,
enabling witnesses to reflect the control-flow context.
"""

from __future__ import annotations
from ir import Expr, BinOp, Var, Const, Neg


def _format_condition_expr(expr: Expr) -> str:
    """Format an IR condition expression as a human-readable string.

    Handles simple comparisons like BinOp(op='>', left=Var('n'), right=Const(3))
    -> 'n > 3'.  Falls back to 'condition at line N' for complex expressions.
    """
    if isinstance(expr, BinOp) and expr.op in {">", ">=", "<", "<=", "==", "~="}:
        left_str = _format_simple(expr.left)
        right_str = _format_simple(expr.right)
        if left_str is not None and right_str is not None:
            return f"{left_str} {expr.op} {right_str}"
    return f"condition at line {expr.line}"


def _format_simple(expr: Expr):
    """Return string for simple atoms (Var, Const, Neg(Const)), else None."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Const):
        v = expr.value
        if v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(expr, Neg) and isinstance(expr.operand, Const):
        v = expr.operand.value
        if v == int(v):
            return str(-int(v))
        return str(-v)
    return None


class PathConstraintStack:
    """Stack of active branch conditions at the current analysis point.

    Each entry is (condition_description: str, branch_taken: bool, line: int).
    """

    def __init__(self):
        self._stack: list = []

    def push(self, condition_expr: Expr, branch_taken: bool, line: int) -> None:
        """Push a branch condition onto the stack."""
        desc = _format_condition_expr(condition_expr)
        self._stack.append((desc, branch_taken, line))

    def pop(self) -> None:
        """Pop the most recent branch condition. Defensive: no-op on empty stack."""
        if self._stack:
            self._stack.pop()

    def snapshot(self) -> tuple:
        """Return a frozen snapshot of the current stack."""
        return tuple(self._stack)
