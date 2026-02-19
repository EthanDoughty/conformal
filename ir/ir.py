# Ethan Doughty
# ir.py
"""Intermediate Representation (IR) for MATLAB.

This module defines a typed, dataclass-based AST for MATLAB programs.
It provides a cleaner representation than the list-based syntax AST.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

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
class StringLit(Expr):
    """String literal ('hello' or "world")."""
    value: str

@dataclass(frozen=True)
class Neg(Expr):
    """Unary negation (-x)."""
    operand: Expr

@dataclass(frozen=True)
class Not(Expr):
    """Logical NOT (~x)."""
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
class FieldAccess(Expr):
    """Struct field access (s.field or s.a.b for nested)."""
    base: Expr
    field: str  # Outermost field in chain (nested as FieldAccess(FieldAccess(...), field))

@dataclass(frozen=True)
class Lambda(Expr):
    """Anonymous function (@(x) x+1 or @(x,y) x+y)."""
    params: List[str]
    body: Expr

@dataclass(frozen=True)
class FuncHandle(Expr):
    """Named function handle (@myFunc)."""
    name: str

@dataclass(frozen=True)
class End(Expr):
    """End keyword in indexing context (resolves to last valid index).

    Examples: c{end}, A(end, :), c{end-1}
    """
    pass

@dataclass(frozen=True)
class Apply(Expr):
    """Unified apply node for runtime call-vs-index disambiguation.

    Represents foo(...) expressions where the decision to treat as a function call
    or indexing operation is deferred to the analyzer based on shape information
    and the presence of colon/range arguments.
    """
    base: Expr
    args: List[IndexArg]

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

# ---- Matrix literals ----

@dataclass(frozen=True)
class MatrixLit(Expr):
    """Matrix literal ([1 2; 3 4])."""
    rows: List[List[Expr]]

@dataclass(frozen=True)
class CellLit(Expr):
    """Cell array literal ({1, 2; 3, 4})."""
    rows: List[List[Expr]]

@dataclass(frozen=True)
class CurlyApply(Expr):
    """Curly-brace content indexing (c{i} or c{i,j}).

    Content indexing: c{i} extracts element (returns element shape, unknown for now).
    Distinct from Apply (parens) which for cells does container indexing.
    """
    base: Expr
    args: List[IndexArg]

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
class StructAssign(Stmt):
    """Struct field assignment (s.field = expr or s.a.b = expr).

    For chained access like s.a.b = expr, fields = ["a", "b"] (flat list).
    """
    base_name: str
    fields: List[str]  # Chain of field names
    expr: Expr

@dataclass(frozen=True)
class CellAssign(Stmt):
    """Cell element assignment (c{i} = expr or c{i,j} = expr).

    Modifies cell variable in-place. Args are indexing arguments (no field chain).
    """
    base_name: str
    args: List[IndexArg]  # Curly index arguments
    expr: Expr

@dataclass(frozen=True)
class IndexAssign(Stmt):
    """Indexed assignment (M(i,j) = expr).

    Assigns to a sub-region of a matrix variable. Does not change the
    variable's overall shape (matrix dimensions preserved).
    """
    base_name: str
    args: List[IndexArg]  # Parenthesized index arguments
    expr: Expr

@dataclass(frozen=True)
class IndexStructAssign(Stmt):
    """Chained indexed struct field assignment (A(i).field = expr, A{i}.field = expr).

    Represents patterns like params(idx).location = val where an indexed
    element is accessed as a struct and a field is assigned.
    """
    base_name: str           # The variable being assigned into ("params")
    index_args: List[IndexArg]  # The indexing arguments ([idx])
    index_kind: str          # "paren" or "curly" -- which type of indexing
    fields: List[str]        # Field chain (["location"] or ["a", "b"] for nested)
    expr: Expr               # RHS expression

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
class FunctionDef(Stmt):
    """Function definition.

    Represents: function [out1, out2] = name(in1, in2)
    Or single return: function result = name(in1)
    Or procedure: function name(in1)
    """
    name: str
    params: List[str]  # Input parameter names
    output_vars: List[str]  # Output variable names (empty for procedures)
    body: List[Stmt]

@dataclass(frozen=True)
class AssignMulti(Stmt):
    """Multiple assignment (destructuring).

    Represents: [a, b, c] = expr
    """
    targets: List[str]  # Variable names to assign
    expr: Expr  # Expression (must evaluate to multiple values)

@dataclass(frozen=True)
class Return(Stmt):
    """Return statement (early exit from function). MATLAB return has no value."""
    pass

@dataclass(frozen=True)
class IfChain(Stmt):
    """If-elseif-else chain.

    Represents: if c1 ... elseif c2 ... elseif c3 ... else ... end
    conditions[0] is the if condition, rest are elseif conditions.
    bodies[0] is the then body, rest are elseif bodies.
    """
    conditions: List[Expr]
    bodies: List[List[Stmt]]
    else_body: List[Stmt]

@dataclass(frozen=True)
class Switch(Stmt):
    """Switch/case statement (MATLAB semantics: no fall-through)."""
    expr: Expr
    cases: List[Tuple[Expr, List[Stmt]]]
    otherwise: List[Stmt]

@dataclass(frozen=True)
class Try(Stmt):
    """Try/catch error handling."""
    try_body: List[Stmt]
    catch_body: List[Stmt]

@dataclass(frozen=True)
class Break(Stmt):
    """Break statement (exit loop)."""
    pass

@dataclass(frozen=True)
class Continue(Stmt):
    """Continue statement (skip to next iteration)."""
    pass

@dataclass(frozen=True)
class Program:
    """Top-level program consisting of statements."""
    body: List[Stmt]