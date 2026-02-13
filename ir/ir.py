# Ethan Doughty
# ir.py
"""Intermediate Representation (IR) for Mini-MATLAB.

This module defines a typed, dataclass-based AST for Mini-MATLAB programs.
It provides a cleaner representation than the list-based syntax AST.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

# ----- Expressions -----

@dataclass(frozen=True)
class Expr:
    """Base class for all expressions."""
    line: int

@dataclass(frozen=True)
class Var(Expr):
    """Variable reference."""
    name: str

@dataclass(frozen=True)
class Const(Expr):
    """Numeric constant."""
    value: float

@dataclass(frozen=True)
class Neg(Expr):
    """Unary negation (-x)."""
    operand: Expr

@dataclass(frozen=True)
class BinOp(Expr):
    """Binary operation (e.g., +, -, *, .*, ==)."""
    op: str
    left: Expr
    right: Expr

@dataclass(frozen=True)
class Transpose(Expr):
    """Matrix transpose (A')."""
    operand: Expr

@dataclass(frozen=True)
class Call(Expr):
    """Function call (e.g., zeros(3,4))."""
    func: Expr
    args: List[Expr]

@dataclass(frozen=True)
class Apply(Expr):
    """Unified apply node for runtime call-vs-index disambiguation.

    Represents foo(...) expressions where the decision to treat as a function call
    or indexing operation is deferred to the analyzer based on shape information
    and the presence of colon/range arguments.
    """
    base: Expr
    args: List  # List[IndexArg]

# ---- Indexing ----

@dataclass(frozen=True)
class IndexArg:
    """Base class for indexing arguments."""
    line: int

@dataclass(frozen=True)
class Colon(IndexArg):
    """Colon indexing (:) - select all elements."""
    pass

@dataclass(frozen=True)
class Range(IndexArg):
    """Range indexing (start:end)."""
    start: Expr
    end: Expr

@dataclass(frozen=True)
class IndexExpr(IndexArg):
    """Single expression index."""
    expr: Expr

@dataclass(frozen=True)
class Index(Expr):
    """Indexing operation (A(i,j), A(:,j), A(2:5,:))."""
    base: Expr
    args: List[IndexArg]

# ---- Matrix literals ----

@dataclass(frozen=True)
class MatrixLit(Expr):
    """Matrix literal ([1 2; 3 4])."""
    rows: List[List[Expr]]

# ---- Statements / Program ----

@dataclass(frozen=True)
class Stmt:
    """Base class for all statements."""
    line: int

@dataclass(frozen=True)
class Assign(Stmt):
    """Assignment statement (x = expr)."""
    name: str
    expr: Expr

@dataclass(frozen=True)
class ExprStmt(Stmt):
    """Expression statement (evaluates but doesn't assign)."""
    expr: Expr

@dataclass(frozen=True)
class If(Stmt):
    """If-else conditional statement."""
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt]

@dataclass(frozen=True)
class While(Stmt):
    """While loop statement."""
    cond: Expr
    body: List[Stmt]

@dataclass(frozen=True)
class For(Stmt):
    """For loop statement (for var = range)."""
    var: str
    it: Expr  # Iterator expression (typically a range like 1:n)
    body: List[Stmt]

@dataclass(frozen=True)
class OpaqueStmt(Stmt):
    """Opaque statement with local-havoc semantics.

    Used to represent unsupported or unknown constructs.
    Sets all target variables to unknown shape during analysis.
    """
    targets: List[str]  # Variable names to havoc
    raw: str = ""  # Optional: original source text

@dataclass(frozen=True)
class Program:
    """Top-level program consisting of statements."""
    body: List[Stmt]