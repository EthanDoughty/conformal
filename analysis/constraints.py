# Ethan Doughty
# constraints.py
"""Constraint tracking and path-sensitive joins for dimension equality."""

from __future__ import annotations
from typing import Optional

from runtime.shapes import Dim, Shape
from runtime.symdim import SymDim
from runtime.env import Env


def _dim_key(d: Dim) -> tuple:
    """Structural sort key for canonicalizing constraint pairs.

    Args:
        d: Dimension (int, SymDim, or None)

    Returns:
        Sort key tuple: (type_tag, value) for stable ordering
        - int: (0, value)
        - SymDim: (1, terms_tuple)
        - None: (2,)
    """
    if isinstance(d, int):
        return (0, d)
    elif isinstance(d, SymDim):
        return (1, d._terms)
    elif d is None:
        return (2,)
    else:
        # Should not reach here
        return (3,)


def record_constraint(ctx, env: Env, dim1: Dim, dim2: Dim, line: int) -> None:
    """Record equality constraint between two dimensions.

    Skips constraints that are:
    - Involving None (unknown can't participate)
    - Trivial (dim1 == dim2)
    - Both concrete ints (already caught by dims_definitely_conflict)
    - Involving pre-bound variables (vars already bound in env)

    Args:
        ctx: AnalysisContext with constraints and constraint_provenance
        env: Current environment for checking pre-bound variables
        dim1: First dimension
        dim2: Second dimension
        line: Source line number (for provenance)
    """
    from analysis.context import AnalysisContext

    # Skip None dims
    if dim1 is None or dim2 is None:
        return

    # Skip trivial equality
    if dim1 == dim2:
        return

    # Skip both-concrete (already caught by dims_definitely_conflict)
    if isinstance(dim1, int) and isinstance(dim2, int):
        return

    # Check for pre-bound vars: if dim is simple var and var is bound in env, skip
    def is_prebound_var(d: Dim) -> bool:
        """Check if dimension is a simple variable that's already bound."""
        if not isinstance(d, SymDim):
            return False
        # Check if SymDim is a simple variable (one term with coeff=1)
        if len(d._terms) != 1:
            return False
        mono, coeff = d._terms[0]
        # Check coefficient is 1
        if coeff != 1:
            return False
        # Check monomial is single variable with exponent 1
        if len(mono) != 1:
            return False
        var_name, exp = mono[0]
        if exp != 1:
            return False
        # Check if var_name is bound in env
        return not env.get(var_name).is_bottom()

    if is_prebound_var(dim1) or is_prebound_var(dim2):
        return

    # Canonicalize: sort by _dim_key to ensure (a, b) and (b, a) are same
    key1 = _dim_key(dim1)
    key2 = _dim_key(dim2)
    if key1 <= key2:
        canonical = (dim1, dim2)
    else:
        canonical = (dim2, dim1)

    # Add to constraints set
    ctx.constraints.add(canonical)

    # Store provenance (overwrite if already exists)
    ctx.constraint_provenance[canonical] = line


def snapshot_constraints(ctx) -> frozenset:
    """Return a copy of current constraint set.

    Args:
        ctx: AnalysisContext with constraints

    Returns:
        Frozen copy of current constraints
    """
    from analysis.context import AnalysisContext
    return frozenset(ctx.constraints)


def join_constraints(baseline: frozenset, branch_sets: list) -> set:
    """Path-sensitive join: keep baseline + constraints added in ALL branches.

    Args:
        baseline: Constraints before branching
        branch_sets: List of constraint sets from each branch (frozenset or set)

    Returns:
        Joined constraint set (baseline + common new constraints)
    """
    # Extract new constraints per branch
    new_per_branch = [set(branch) - set(baseline) for branch in branch_sets]

    # Intersection of all new constraints (only if all branches added them)
    if not new_per_branch:
        common_new = set()
    else:
        common_new = set(new_per_branch[0])
        for new_set in new_per_branch[1:]:
            common_new &= new_set

    # Return baseline + common new
    return set(baseline) | common_new
