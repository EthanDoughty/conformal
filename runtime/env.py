# Ethan Doughty
# env.py
"""Environment for tracking variable shapes during analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from runtime.shapes import Shape, join_shape, widen_shape


@dataclass
class Env:
    """Mapping from variable names to their inferred shapes."""
    bindings: Dict[str, Shape] = field(default_factory=dict)

    def copy(self) -> "Env":
        """Create a shallow copy of this environment."""
        return Env(bindings=self.bindings.copy())

    def get(self, name: str) -> Shape:
        """Get the shape of a variable, or bottom if not found."""
        return self.bindings.get(name, Shape.bottom())

    def set(self, name: str, shape: Shape) -> None:
        """Set the shape of a variable."""
        self.bindings[name] = shape

    def __repr__(self) -> str:
        """String representation showing all bindings."""
        parts = [f"{var_name}: {shape}" for var_name, shape in self.bindings.items()]
        return "Env{" + ", ".join(parts) + "}"


def join_env(env1: Env, env2: Env) -> Env:
    """Merge two environments by joining shapes pointwise.

    Args:
        env1: First environment (e.g., from then-branch)
        env2: Second environment (e.g., from else-branch)

    Returns:
        Merged environment with joined shapes for each variable
    """
    result = Env()
    all_vars = sorted(set(env1.bindings.keys()) | set(env2.bindings.keys()))
    for var_name in all_vars:
        shape1 = env1.get(var_name)
        shape2 = env2.get(var_name)
        result.set(var_name, join_shape(shape1, shape2))
    return result


def widen_env(env1: Env, env2: Env) -> Env:
    """Widen two environments pointwise. Used for both widening and post-loop join.

    Used in fixpoint loop analysis to accelerate convergence by widening
    conflicting dimensions to None (unknown). Also used for post-loop join
    to model "loop may not execute" semantics.

    Args:
        env1: First environment (old/pre-loop state)
        env2: Second environment (new/post-iteration state)

    Returns:
        Widened environment with pointwise widen_shape applied
    """
    result = Env()
    all_vars = sorted(set(env1.bindings.keys()) | set(env2.bindings.keys()))
    for var_name in all_vars:
        shape1 = env1.get(var_name)    # returns unknown if unbound
        shape2 = env2.get(var_name)
        result.set(var_name, widen_shape(shape1, shape2))
    return result