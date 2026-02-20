# Ethan Doughty
# env.py
"""Environment for tracking variable shapes during analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from runtime.shapes import Shape, join_shape, widen_shape


@dataclass
class Env:
    """Mapping from variable names to their inferred shapes.

    Supports optional parent-pointer scope chains for nested scoping.
    get() walks the chain; set() writes to local scope only.
    """
    bindings: Dict[str, Shape] = field(default_factory=dict)
    dim_aliases: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[Env] = None

    def copy(self) -> Env:
        """Shallow copy of local scope, sharing the same parent."""
        return Env(bindings=self.bindings.copy(),
                   dim_aliases=self.dim_aliases.copy(),
                   parent=self.parent)

    def get(self, name: str) -> Shape:
        """Get shape by name, walking parent chain if not local."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        return Shape.bottom()

    def set(self, name: str, shape: Shape) -> None:
        """Set shape in local scope."""
        self.bindings[name] = shape

    def has_local(self, name: str) -> bool:
        """Check if name is bound in local scope (not parent)."""
        return name in self.bindings

    def __contains__(self, name: str) -> bool:
        """Check if name is bound anywhere in the scope chain."""
        if name in self.bindings:
            return True
        return self.parent is not None and name in self.parent

    def push_scope(self) -> Env:
        """Create a child scope with this env as parent."""
        return Env(parent=self)

    def replace_local(self, other: Env) -> None:
        """Replace local bindings and aliases from another env."""
        self.bindings = other.bindings
        self.dim_aliases = other.dim_aliases

    def local_bindings_equal(self, other: Env) -> bool:
        """Compare local bindings only (not parent chain)."""
        return self.bindings == other.bindings

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
    result = Env(parent=env1.parent)
    all_vars = sorted(set(env1.bindings.keys()) | set(env2.bindings.keys()))
    for var_name in all_vars:
        shape1 = env1.bindings.get(var_name, Shape.bottom())
        shape2 = env2.bindings.get(var_name, Shape.bottom())
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
    result = Env(parent=env1.parent)
    all_vars = sorted(set(env1.bindings.keys()) | set(env2.bindings.keys()))
    for var_name in all_vars:
        shape1 = env1.bindings.get(var_name, Shape.bottom())
        shape2 = env2.bindings.get(var_name, Shape.bottom())
        result.set(var_name, widen_shape(shape1, shape2))
    return result