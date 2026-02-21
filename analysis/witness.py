# Ethan Doughty
# witness.py
"""Under-approximate witness generation for Conformal.

Produces concrete variable assignments that prove a warning is a real bug.
Every returned Witness is verified: dim_a_concrete != dim_b_concrete.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import itertools

from runtime.symdim import SymDim

if TYPE_CHECKING:
    from runtime.shapes import Dim


@dataclass(frozen=True)
class Witness:
    """Concrete proof that a warning is a real bug."""
    assignments: tuple       # ((var_name, value), ...) only conflict-relevant vars
    dim_a_concrete: int      # dim_a evaluated under witness
    dim_b_concrete: int      # dim_b evaluated under witness
    explanation: str         # Human-readable: "n=3, m=5 -> dims 3 != 5"


@dataclass(frozen=True)
class ConflictSite:
    """Recorded dimension conflict at a warning emission point."""
    dim_a: object            # Dim (int, SymDim, or None)
    dim_b: object            # Dim (int, SymDim, or None)
    line: int
    warning_code: str        # e.g., "W_INNER_DIM_MISMATCH"
    constraints_snapshot: frozenset   # ctx.constraints at time of conflict
    scalar_bindings_snapshot: tuple   # ctx.scalar_bindings items at time of conflict
    value_ranges_snapshot: tuple      # ctx.value_ranges items at time of conflict


def _collect_relevant_vars(site: ConflictSite) -> set:
    """Collect variable names from dim_a, dim_b, and equality-linked constraints."""
    relevant = set()

    def add_dim_vars(d):
        if isinstance(d, SymDim):
            relevant.update(d.variables())

    add_dim_vars(site.dim_a)
    add_dim_vars(site.dim_b)

    # Walk equality constraints: if a var in relevant appears in a constraint,
    # add the other dim's vars too (one hop of transitivity).
    changed = True
    while changed:
        changed = False
        for d1, d2 in site.constraints_snapshot:
            vars1 = d1.variables() if isinstance(d1, SymDim) else set()
            vars2 = d2.variables() if isinstance(d2, SymDim) else set()
            if vars1 & relevant:
                before = len(relevant)
                relevant.update(vars2)
                if len(relevant) > before:
                    changed = True
            if vars2 & relevant:
                before = len(relevant)
                relevant.update(vars1)
                if len(relevant) > before:
                    changed = True

    return relevant


def _eval_dim(d, bindings: dict) -> Optional[int]:
    """Evaluate a Dim (int, SymDim, or None) under bindings."""
    if d is None:
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, SymDim):
        return d.evaluate(bindings)
    return None


def _constraints_satisfied(site: ConflictSite, bindings: dict) -> bool:
    """Check that all snapshot equality constraints are satisfied under bindings."""
    for d1, d2 in site.constraints_snapshot:
        v1 = _eval_dim(d1, bindings)
        v2 = _eval_dim(d2, bindings)
        # Only check if both sides are fully evaluable
        if v1 is not None and v2 is not None:
            if v1 != v2:
                return False
    return True


def _find_satisfying_assignment(
    site: ConflictSite,
    relevant_vars: set,
    scalar_bindings: dict,
) -> Optional[Witness]:
    """Enumerate assignments and verify witness.

    Strategy:
    1. Pre-fill from scalar_bindings (concrete known values).
    2. For remaining free vars, use interval bounds from value_ranges_snapshot
       or default [0..10], clamped to 20 values max.
    3. Enumerate all combinations, verify constraints + mismatch.
    """
    # Build value_ranges dict from snapshot
    value_ranges = dict(site.value_ranges_snapshot)

    # Pre-fill from scalar_bindings (these are known concrete values)
    pre_bindings = {k: v for k, v in scalar_bindings.items() if k in relevant_vars}

    # Determine free vars (relevant but not pre-bound)
    free_vars = sorted(relevant_vars - set(pre_bindings.keys()))

    # Bail out for too many free variables (> 8, per spec)
    if len(free_vars) > 8:
        return None

    # Build candidate value ranges for each free var
    def candidate_range(var: str) -> List[int]:
        if var in value_ranges:
            iv = value_ranges[var]
            lo = iv[0]
            hi = iv[1]
            # lo/hi may be int or SymDim; only use concrete int bounds
            if not isinstance(lo, int):
                lo = 0
            if not isinstance(hi, int):
                hi = 10
            lo = max(0, lo)
            hi = min(hi, lo + 20)
            return list(range(lo, hi + 1))
        return list(range(0, 11))

    candidates = [candidate_range(v) for v in free_vars]

    for combo in itertools.product(*candidates):
        bindings = dict(pre_bindings)
        for var, val in zip(free_vars, combo):
            bindings[var] = val

        # Evaluate dims
        a = _eval_dim(site.dim_a, bindings)
        b = _eval_dim(site.dim_b, bindings)

        # Skip if either dim is non-integer or negative
        if a is None or b is None:
            continue
        if a < 0 or b < 0:
            continue

        # Verify the conflict is real
        if a == b:
            continue

        # Verify all constraints are satisfied
        if not _constraints_satisfied(site, bindings):
            continue

        # Build assignments tuple (only relevant vars that were actually used)
        used = sorted(
            (var, val) for var, val in bindings.items() if var in relevant_vars
        )
        assign_parts = ", ".join(f"{var}={val}" for var, val in used)
        explanation = f"{assign_parts} -> dims {a} != {b}"

        return Witness(
            assignments=tuple(used),
            dim_a_concrete=a,
            dim_b_concrete=b,
            explanation=explanation,
        )

    return None


def attempt_witness(site: ConflictSite) -> Optional[Witness]:
    """Try to construct a concrete witness for a ConflictSite.

    Returns:
        Witness if a valid assignment found, None on failure/degradation.
    """
    dim_a = site.dim_a
    dim_b = site.dim_b

    # Case: either dim is None (unknown) — cannot construct witness
    if dim_a is None or dim_b is None:
        return None

    # Bail on quadratic+ terms (degree > 1 in any monomial)
    def max_degree(d) -> int:
        if not isinstance(d, SymDim):
            return 0
        max_deg = 0
        for mono, _ in d._terms:
            deg = sum(exp for _, exp in mono)
            if deg > max_deg:
                max_deg = deg
        return max_deg

    if max_degree(dim_a) > 1 or max_degree(dim_b) > 1:
        return None

    # Case 1: Both dims are concrete ints — trivial witness
    if isinstance(dim_a, int) and isinstance(dim_b, int):
        if dim_a == dim_b:
            return None  # Not actually a conflict
        return Witness(
            assignments=(),
            dim_a_concrete=dim_a,
            dim_b_concrete=dim_b,
            explanation=f"dims {dim_a} != {dim_b}",
        )

    # Case 2: Symbolic dims — enumerate
    relevant_vars = _collect_relevant_vars(site)

    # Also bail if too many vars after linking
    if len(relevant_vars) > 8:
        return None

    scalar_bindings = dict(site.scalar_bindings_snapshot)

    # Check if scalar_bindings alone already ground everything
    a = _eval_dim(dim_a, scalar_bindings)
    b = _eval_dim(dim_b, scalar_bindings)
    if a is not None and b is not None and a >= 0 and b >= 0 and a != b:
        if _constraints_satisfied(site, scalar_bindings):
            used = sorted(
                (var, val) for var, val in scalar_bindings.items()
                if var in relevant_vars
            )
            assign_parts = ", ".join(f"{var}={val}" for var, val in used)
            explanation = f"{assign_parts} -> dims {a} != {b}"
            return Witness(
                assignments=tuple(used),
                dim_a_concrete=a,
                dim_b_concrete=b,
                explanation=explanation,
            )

    return _find_satisfying_assignment(site, relevant_vars, scalar_bindings)


def generate_witnesses(conflict_sites: List[ConflictSite]) -> Dict[tuple, Witness]:
    """Batch-process conflict sites and return witnesses keyed by (line, code).

    Args:
        conflict_sites: List of ConflictSite records from analysis.

    Returns:
        Dict mapping (line, warning_code) -> Witness for each site with a witness.
    """
    result = {}
    for site in conflict_sites:
        key = (site.line, site.warning_code)
        if key in result:
            continue  # Already have a witness for this (line, code)
        w = attempt_witness(site)
        if w is not None:
            result[key] = w
    return result
