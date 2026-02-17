# Ethan Doughty
# intervals.py
"""Integer interval abstract domain for value range analysis."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass(frozen=True)
class Interval:
    """Integer interval [lo, hi] with optional bounds.

    lo=None means -infinity, hi=None means +infinity
    Invariant: if both non-None, lo <= hi
    """
    lo: Optional[int]
    hi: Optional[int]

    def __post_init__(self):
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"Invalid interval: lo={self.lo} > hi={self.hi}")


def join_interval(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Convex hull of two intervals (lattice join).

    None represents absence of information (treat as absorbing).
    """
    if a is None:
        return b
    if b is None:
        return a

    # Compute min of lower bounds (None = -infinity)
    if a.lo is None or b.lo is None:
        new_lo = None
    else:
        new_lo = min(a.lo, b.lo)

    # Compute max of upper bounds (None = +infinity)
    if a.hi is None or b.hi is None:
        new_hi = None
    else:
        new_hi = max(a.hi, b.hi)

    return Interval(new_lo, new_hi)


def widen_interval(old: Interval, new: Interval) -> Interval:
    """Widening operator: push bounds to infinity when they move."""
    # Lower bound: if new.lo < old.lo, widen to -infinity
    if new.lo is not None and old.lo is not None and new.lo < old.lo:
        lo = None
    else:
        lo = old.lo

    # Upper bound: if new.hi > old.hi, widen to +infinity
    if new.hi is not None and old.hi is not None and new.hi > old.hi:
        hi = None
    else:
        hi = old.hi

    return Interval(lo, hi)


def meet_interval(a: Interval, b: Interval) -> Optional[Interval]:
    """Intersection of two intervals (lattice meet).

    Returns None if intervals don't overlap.
    """
    # Compute max of lower bounds
    if a.lo is None:
        new_lo = b.lo
    elif b.lo is None:
        new_lo = a.lo
    else:
        new_lo = max(a.lo, b.lo)

    # Compute min of upper bounds
    if a.hi is None:
        new_hi = b.hi
    elif b.hi is None:
        new_hi = a.hi
    else:
        new_hi = min(a.hi, b.hi)

    # Check if valid (non-empty interval)
    if new_lo is not None and new_hi is not None and new_lo > new_hi:
        return None  # Empty interval

    return Interval(new_lo, new_hi)


def interval_add(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    """Interval addition: [a,b] + [c,d] = [a+c, b+d]."""
    if a is None or b is None:
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
    """Interval subtraction: [a,b] - [c,d] = [a-d, b-c]."""
    if a is None or b is None:
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
    """Interval multiplication: standard 4-product min/max."""
    if a is None or b is None:
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
    """Interval negation: -[a,b] = [-b, -a]."""
    if a is None:
        return None

    # Negate and swap bounds
    new_lo = -a.hi if a.hi is not None else None
    new_hi = -a.lo if a.lo is not None else None

    return Interval(new_lo, new_hi)


def interval_is_exactly_zero(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely exactly zero."""
    if iv is None:
        return False
    return iv.lo == 0 and iv.hi == 0


def interval_definitely_positive(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely > 0."""
    if iv is None:
        return False
    return iv.lo is not None and iv.lo > 0


def interval_definitely_nonpositive(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely <= 0."""
    if iv is None:
        return False
    return iv.hi is not None and iv.hi <= 0


def interval_definitely_negative(iv: Optional[Interval]) -> bool:
    """Check if interval is definitely < 0 (strictly negative)."""
    if iv is None:
        return False
    return iv.hi is not None and iv.hi < 0


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
