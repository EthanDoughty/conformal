# Ethan Doughty
# context.py
"""Analysis context, function signatures, and control-flow exceptions."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Set, TYPE_CHECKING

from ir import Stmt
from analysis.path_constraints import PathConstraintStack

if TYPE_CHECKING:
    from analysis.workspace import ExternalSignature
    from analysis.intervals import Interval


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
class CallContext:
    """Function/lambda dispatch state."""
    function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)
    analyzing_functions: Set[str] = field(default_factory=set)
    analysis_cache: Dict[tuple, tuple] = field(default_factory=dict)
    fixpoint: bool = False
    _lambda_metadata: Dict[int, tuple] = field(default_factory=dict)  # lambda_id -> (params, body, closure_env)
    _handle_registry: Dict[int, str] = field(default_factory=dict)  # handle_id -> function_name
    analyzing_lambdas: Set[int] = field(default_factory=set)
    _next_lambda_id: int = 0
    nested_function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)


@dataclass
class ConstraintContext:
    """Path-sensitive constraint and interval state."""
    constraints: set = field(default_factory=set)  # Set of (dim1, dim2) tuples (canonicalized)
    constraint_provenance: dict = field(default_factory=dict)  # (dim1, dim2) -> source_line
    scalar_bindings: Dict[str, int] = field(default_factory=dict)  # var_name -> concrete_value
    value_ranges: Dict[str, 'Interval'] = field(default_factory=dict)  # var_name -> integer interval
    conflict_sites: list = field(default_factory=list)  # List[ConflictSite] accumulated globally
    dim_provenance: dict = field(default_factory=dict)  # (var_name, "rows"|"cols") -> Dim
    path_constraints: PathConstraintStack = field(default_factory=PathConstraintStack)


@dataclass
class WorkspaceContext:
    """Cross-file analysis state."""
    external_functions: Dict[str, 'ExternalSignature'] = field(default_factory=dict)
    analyzing_external: Set[str] = field(default_factory=set)


@dataclass
class AnalysisContext:
    """Threaded analysis state (replaces loose parameters).

    Fields are grouped into three focused sub-objects:
    - call: function/lambda dispatch (CallContext)
    - cst:  constraint and interval state (ConstraintContext)
    - ws:   cross-file workspace state (WorkspaceContext)
    """
    call: CallContext = field(default_factory=CallContext)
    cst: ConstraintContext = field(default_factory=ConstraintContext)
    ws: WorkspaceContext = field(default_factory=WorkspaceContext)

    @contextmanager
    def snapshot_scope(self):
        """Save and restore scope-sensitive fields around function/lambda body analysis."""
        saved_constraints = set(self.cst.constraints)
        saved_provenance = dict(self.cst.constraint_provenance)
        saved_scalars = dict(self.cst.scalar_bindings)
        saved_ranges = dict(self.cst.value_ranges)
        saved_nested = dict(self.call.nested_function_registry)
        try:
            yield
        finally:
            self.cst.constraints = saved_constraints
            self.cst.constraint_provenance = saved_provenance
            self.cst.scalar_bindings = saved_scalars
            self.cst.value_ranges = saved_ranges
            self.call.nested_function_registry = saved_nested
