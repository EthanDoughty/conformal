# Ethan Doughty
# env.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from runtime.shapes import Shape, join_shape


@dataclass
class Env:
    """Mapping from variable names to Shapes"""
    bindings: Dict[str, Shape] = field(default_factory=dict)

    def copy(self) -> "Env":
        return Env(bindings=self.bindings.copy())

    def get(self, name: str) -> Shape:
        return self.bindings.get(name, Shape.unknown())

    def set(self, name: str, shape: Shape) -> None:
        self.bindings[name] = shape

    def __repr__(self) -> str:
        parts = [f"{k}: {v}" for k, v in self.bindings.items()]
        return "Env{" + ", ".join(parts) + "}"


def join_env(e1: Env, e2: Env) -> Env:
    """Pointwise join of two environments"""
    result = Env()
    keys = set(e1.bindings.keys()) | set(e2.bindings.keys())
    for k in keys:
        s1 = e1.get(k)
        s2 = e2.get(k)
        result.set(k, join_shape(s1, s2))
    return result