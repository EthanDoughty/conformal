# Ethan Doughty
# intervals.py
"""Integer interval abstract domain for value range analysis."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple, TYPE_CHECKING

from runtime.symdim import SymDim

if TYPE_CHECKING:
    from ir import Expr
    from runtime.env import Env
    from analysis.context import AnalysisContext

# Bound type: int, SymDim, or None (unbounded)
Bound = Union[int, SymDim, None]


@dataclass(frozen=True)
class Interval:
    """Integer interval [lo, hi] with optional symbolic bounds.

    lo=None means -infinity, hi=None means +infinity
    Bounds can be int (concrete) or SymDim (symbolic)
    Invariant: if both concrete (int), lo <= hi
    """
    lo: Bound
    hi: Bound

    def __post_init__(self):
        # Guard validation: only check if both bounds are concrete ints (B4)
        if isinstance(self.lo, int) and isinstance(self.hi, int) and self.lo > self.hi:
            raise ValueError(f"Invalid interval: lo={self.lo} > hi={self.hi}")


def join_interval(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Convex hull of two intervals (lattice join).

    None represents absence of information (treat as absorbing).
    Symbolic bounds handled conservatively.
    """
    if a is None:
        return b
    if b is None:
        return a

    # Compute min of lower bounds (None = -infinity)
    if a.lo is None or b.lo is None:
        new_lo = None
    elif isinstance(a.lo, int) and isinstance(b.lo, int):
        new_lo = min(a.lo, b.lo)
    elif a.lo == b.lo:  # Structural equality (works for SymDim)
        new_lo = a.lo
    else:
        # Can't compare: widen to -inf (sound)
        new_lo = None

    # Compute max of upper bounds (None = +infinity)
    if a.hi is None or b.hi is None:
        new_hi = None
    elif isinstance(a.hi, int) and isinstance(b.hi, int):
        new_hi = max(a.hi, b.hi)
    elif a.hi == b.hi:  # Structural equality
        new_hi = a.hi
    else:
        # Can't compare: widen to +inf (sound)
        new_hi = None

    return Interval(new_lo, new_hi)


def widen_interval(old: Interval, new: Interval) -> Interval:
    """Widening operator: push bounds to infinity when they move."""
    # Lower bound: if new.lo < old.lo, widen to -infinity (B2: guard SymDim comparisons)
    if new.lo is not None and old.lo is not None:
        if isinstance(new.lo, int) and isinstance(old.lo, int):
            if new.lo < old.lo:
                lo = None
            else:
                lo = old.lo
        else:
            # Symbolic bound: can't prove it didn't move, widen to None (sound)
            lo = None
    else:
        lo = old.lo

    # Upper bound: if new.hi > old.hi, widen to +infinity (B2: guard SymDim comparisons)
    if new.hi is not None and old.hi is not None:
        if isinstance(new.hi, int) and isinstance(old.hi, int):
            if new.hi > old.hi:
                hi = None
            else:
                hi = old.hi
        else:
            # Symbolic bound: can't prove it didn't move, widen to None (sound)
            hi = None
    else:
        hi = old.hi

    return Interval(lo, hi)


def meet_interval(a: Interval, b: Interval) -> Optional[Interval]:
    """Intersection of two intervals (lattice meet).

    Returns None if intervals don't overlap. (B3: SymDim-safe)
    """
    # Compute max of lower bounds
    if a.lo is None:
        new_lo = b.lo
    elif b.lo is None:
        new_lo = a.lo
    elif isinstance(a.lo, int) and isinstance(b.lo, int):
        new_lo = max(a.lo, b.lo)
    elif a.lo == b.lo:  # Structural equality (works for SymDim)
        new_lo = a.lo
    else:
        # Can't compare: use more constrained bound or None (conservative)
        new_lo = None

    # Compute min of upper bounds
    if a.hi is None:
        new_hi = b.hi
    elif b.hi is None:
        new_hi = a.hi
    elif isinstance(a.hi, int) and isinstance(b.hi, int):
        new_hi = min(a.hi, b.hi)
    elif a.hi == b.hi:  # Structural equality
        new_hi = a.hi
    else:
        # Can't compare: use more constrained bound or None (conservative)
        new_hi = None

    # Check if valid (non-empty interval) — guard comparisons (B3)
    if new_lo is not None and new_hi is not None:
        if isinstance(new_lo, int) and isinstance(new_hi, int):
            if new_lo > new_hi:
                return None  # Empty interval
        # else: at least one symbolic, can't prove empty, assume non-empty (sound)

    return Interval(new_lo, new_hi)


def _is_concrete_bound(b: Bound) -> bool:
    """Check if a bound is a concrete int (not SymDim)."""
    return b is None or isinstance(b, int)


def interval_add(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Interval addition: [a,b] + [c,d] = [a+c, b+d].

    Returns None (top) if any bound is symbolic (no downstream benefit).
    """
    if a is None or b is None:
        return None

    # Short-circuit on symbolic bounds (B5 pattern)
    if not all(_is_concrete_bound(x) for x in [a.lo, a.hi, b.lo, b.hi]):
        return None

    # Add lower bounds
    if a.lo is None or b.lo is None:
        new_lo = None
    else:
        new_lo = a.lo + b.lo

    # Add upper bounds
    if a.hi is None or b.hi is None:
        new_hi = None
    else:
        new_hi = a.hi + b.hi

    return Interval(new_lo, new_hi)


def interval_sub(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Interval subtraction: [a,b] - [c,d] = [a-d, b-c].

    Returns None (top) if any bound is symbolic.
    """
    if a is None or b is None:
        return None

    # Short-circuit on symbolic bounds
    if not all(_is_concrete_bound(x) for x in [a.lo, a.hi, b.lo, b.hi]):
        return None

    # Subtract: lo - hi (reversed)
    if a.lo is None or b.hi is None:
        new_lo = None
    else:
        new_lo = a.lo - b.hi

    # Subtract: hi - lo (reversed)
    if a.hi is None or b.lo is None:
        new_hi = None
    else:
        new_hi = a.hi - b.lo

    return Interval(new_lo, new_hi)


def interval_mul(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Interval multiplication: standard 4-product min/max.

    Returns None (top) if any bound is symbolic.
    """
    if a is None or b is None:
        return None

    # Short-circuit on symbolic bounds
    if not all(_is_concrete_bound(x) for x in [a.lo, a.hi, b.lo, b.hi]):
        return None

    # Extract bounds (treat None as unbounded)
    a_lo = a.lo if a.lo is not None else float('-inf')
    a_hi = a.hi if a.hi is not None else float('+inf')
    b_lo = b.lo if b.lo is not None else float('-inf')
    b_hi = b.hi if b.hi is not None else float('+inf')

    # Compute all 4 products (if finite)
    products = []
    for x in [a_lo, a_hi]:
        for y in [b_lo, b_hi]:
            if abs(x) != float('inf') and abs(y) != float('inf'):
                products.append(x * y)

    # If all products involve infinity, return top
    if not products:
        return Interval(None, None)

    new_lo = int(min(products))
    new_hi = int(max(products))

    return Interval(new_lo, new_hi)


def interval_neg(a: Optional[Interval]) -> Optional[Interval]:
    """Interval negation: -[a,b] = [-b, -a].

    Returns None (top) if any bound is symbolic (B5).
    """
    if a is None:
        return None

    # Short-circuit on symbolic bounds
    if not all(_is_concrete_bound(x) for x in [a.lo, a.hi]):
        return None

    # Negate and swap bounds
    new_lo = -a.hi if a.hi is not None else None
    new_hi = -a.lo if a.lo is not None else None

    return Interval(new_lo, new_hi)


def interval_is_exactly_zero(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely exactly zero.

    Symbolic bounds: can't prove equality, return False (sound).
    """
    if iv is None:
        return False
    # Guard symbolic bounds
    if not isinstance(iv.lo, int) or not isinstance(iv.hi, int):
        return False
    return iv.lo == 0 and iv.hi == 0


def interval_definitely_positive(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely > 0.

    Symbolic bounds: can't prove, return False (sound).
    """
    if iv is None:
        return False
    if iv.lo is None:
        return False
    # Guard symbolic bounds
    if not isinstance(iv.lo, int):
        return False
    return iv.lo > 0


def interval_definitely_nonpositive(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely <= 0.

    Symbolic bounds: can't prove, return False (sound).
    """
    if iv is None:
        return False
    if iv.hi is None:
        return False
    # Guard symbolic bounds
    if not isinstance(iv.hi, int):
        return False
    return iv.hi <= 0


def interval_definitely_negative(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely < 0 (strictly negative).

    Symbolic bounds: can't prove, return False (sound).
    """
    if iv is None:
        return False
    if iv.hi is None:
        return False
    # Guard symbolic bounds
    if not isinstance(iv.hi, int):
        return False
    return iv.hi < 0


def join_value_ranges(baseline: Dict[str, Interval], branch_ranges: List[Dict[str, Interval]]) -> Dict[str, Interval]:
    """Join value_ranges dicts across branches (convex hull per variable).

    Variables absent from a branch are treated as top (no information).
    Only variables present in at least one branch appear in the result.

    Args:
        baseline: Pre-branch value_ranges (not directly used, here for signature consistency)
        branch_ranges: List of per-branch value_ranges dicts

    Returns:
        Joined value_ranges dict
    """
    all_vars = set()
    for br in branch_ranges:
        all_vars.update(br.keys())

    result = {}
    for var in all_vars:
        intervals = [br.get(var) for br in branch_ranges if var in br]
        if intervals:
            joined = intervals[0]
            for iv in intervals[1:]:
                joined = join_interval(joined, iv)
            result[var] = joined

    return result


# ===== Conditional Interval Refinement (v1.8.0) =====


def negate_comparison_op(op: str) -> str:
    """Negate a comparison operator for else-branch refinement.

    | Original | Negated |
    |----------|---------|
    | >        | <=      |
    | >=       | <       |
    | <        | >=      |
    | <=       | >       |
    | ==       | ~=      |
    | ~=       | ==      |
    """
    negations = {
        '>': '<=',
        '>=': '<',
        '<': '>=',
        '<=': '>',
        '==': '~=',
        '~=': '==',
    }
    return negations.get(op, op)


def interval_from_comparison(op: str, bound: Bound) -> Optional[Interval]:
    """Convert a comparison to a guard interval.

    Uses add_dim for bound arithmetic (works for int and SymDim).

    Args:
        op: Comparison operator (>, >=, <, <=, ==, ~=)
        bound: Right-hand side bound (int, SymDim, or None)

    Returns:
        Guard interval, or None if not representable
    """
    from runtime.shapes import add_dim

    if bound is None:
        return None

    if op == '>':
        # x > c means x >= c+1 (integer semantics)
        return Interval(add_dim(bound, 1), None)
    elif op == '>=':
        return Interval(bound, None)
    elif op == '<':
        # x < c means x <= c-1
        return Interval(None, add_dim(bound, -1))
    elif op == '<=':
        return Interval(None, bound)
    elif op == '==':
        return Interval(bound, bound)
    elif op == '~=':
        # Not-equal: can't represent as interval (excludes single point)
        return None
    else:
        return None


def _get_expr_bound(expr: 'Expr', env: 'Env', ctx: 'AnalysisContext') -> Bound:
    """Extract a bound from an expression.

    Tries Const → int, Var with exact interval → int, expr_to_dim_ir → SymDim.
    Returns None if not extractable.

    Args:
        expr: Expression to extract bound from
        env: Current environment
        ctx: Analysis context

    Returns:
        int, SymDim, or None
    """
    from ir import Const, Var, BinOp
    from analysis.dim_extract import expr_to_dim_ir

    # Try constant extraction
    if isinstance(expr, Const):
        if isinstance(expr.value, (int, float)) and expr.value == int(expr.value):
            return int(expr.value)
        return None

    # Try exact interval from variable
    if isinstance(expr, Var):
        interval = ctx.cst.value_ranges.get(expr.name)
        if interval is not None and interval.lo == interval.hi:
            # Exact value (works for both int and SymDim via __eq__)
            return interval.lo
        # Fall through to dimension extraction

    # Try dimension extraction (returns int or SymDim)
    dim = expr_to_dim_ir(expr, env)
    return dim


def extract_condition_refinements(cond: 'Expr', env: 'Env', ctx: 'AnalysisContext') -> List[Tuple[str, str, Bound]]:
    """Extract interval refinements from a branch condition.

    Pattern-matches:
    - BinOp(op in {>, >=, <, <=, ==, ~=}, left=Var, right=<expr>) → [(var, op, bound)]
    - BinOp(op, left=<expr>, right=Var) → [(var, flipped_op, bound)] (flipped)
    - BinOp(op='&&', left, right) → union of both sides (conjunction)
    - BinOp(op='&', left, right) → union of both sides (element-wise and)
    - BinOp(op='||', ...) → [] (disjunction: no safe refinement)
    - BinOp(op='|', ...) → [] (element-wise or: no safe refinement)

    Args:
        cond: Condition expression
        env: Current environment
        ctx: Analysis context

    Returns:
        List of (var_name, comparison_op, bound) tuples
    """
    from ir import Var, BinOp

    if not isinstance(cond, BinOp):
        return []

    op = cond.op

    # Conjunction: merge refinements from both sides
    if op in ('&&', '&'):
        left_refinements = extract_condition_refinements(cond.left, env, ctx)
        right_refinements = extract_condition_refinements(cond.right, env, ctx)
        return left_refinements + right_refinements

    # Disjunction: no safe single-variable refinement
    if op in ('||', '|'):
        return []

    # Comparison operators
    if op not in ('>', '>=', '<', '<=', '==', '~='):
        return []

    # Pattern match: left=Var, right=<expr>
    if isinstance(cond.left, Var):
        var_name = cond.left.name
        bound = _get_expr_bound(cond.right, env, ctx)
        if bound is not None:
            return [(var_name, op, bound)]

    # Pattern match: left=<expr>, right=Var (flip operator)
    if isinstance(cond.right, Var):
        var_name = cond.right.name
        bound = _get_expr_bound(cond.left, env, ctx)
        if bound is not None:
            # Flip operator: x < 5 is same as 5 > x
            flip_map = {
                '>': '<',
                '>=': '<=',
                '<': '>',
                '<=': '>=',
                '==': '==',
                '~=': '~=',
            }
            flipped_op = flip_map.get(op, op)
            return [(var_name, flipped_op, bound)]

    return []


def _apply_refinements(ctx: 'AnalysisContext', refinements: List[Tuple[str, str, Bound]], negate: bool = False):
    """Apply interval refinements to ctx.value_ranges.

    Modifies ctx.value_ranges in place.

    Args:
        ctx: Analysis context
        refinements: List of (var_name, op, bound) tuples
        negate: If True, negate the comparison operator
    """
    for var_name, op, bound in refinements:
        current = ctx.cst.value_ranges.get(var_name, Interval(None, None))
        if negate:
            op = negate_comparison_op(op)
        guard = interval_from_comparison(op, bound)
        if guard is not None:
            refined = meet_interval(current, guard)
            if refined is not None:
                ctx.cst.value_ranges[var_name] = refined
            else:
                # Meet is empty: branch is dead code. Use guard interval to
                # prevent false positives inside unreachable branches.
                ctx.cst.value_ranges[var_name] = guard
