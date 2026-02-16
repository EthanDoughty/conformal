# Ethan Doughty
# context.py
"""Analysis context, function signatures, and control-flow exceptions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set

from ir import Stmt


@dataclass
class FunctionSignature:
    """Registered user-defined function signature."""
    name: str
    params: List[str]
    output_vars: List[str]
    body: List[Stmt]

class EarlyReturn(Exception):
    """Raised by return statement to exit function body analysis."""
    pass

class EarlyBreak(Exception):
    """Raised by break statement to exit loop."""
    pass

class EarlyContinue(Exception):
    """Raised by continue statement to skip to next iteration."""
    pass

@dataclass
class AnalysisContext:
    """Threaded analysis state (replaces loose parameters)."""
    function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)
    analyzing_functions: Set[str] = field(default_factory=set)
    analysis_cache: Dict[tuple, tuple] = field(default_factory=dict)
    fixpoint: bool = False
    _lambda_metadata: Dict[int, tuple] = field(default_factory=dict)  # lambda_id -> (params, body, closure_env)
    _handle_registry: Dict[int, str] = field(default_factory=dict)  # handle_id -> function_name
    analyzing_lambdas: Set[int] = field(default_factory=set)
    _next_lambda_id: int = 0
    constraints: set = field(default_factory=set)  # Set of (dim1, dim2) tuples (canonicalized)
    constraint_provenance: dict = field(default_factory=dict)  # (dim1, dim2) -> source_line
    scalar_bindings: Dict[str, int] = field(default_factory=dict)  # var_name -> concrete_value (for constraint validation)
