# Ethan Doughty
# analysis_ast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Literal

# ----- Expressions -----

@dataclass(frozen=True)
class Expr:
    line: int

@dataclass(frozen=True)
class Var(Expr):
    name: str

@dataclass(frozen=True)
class Const(Expr):
    value: float

@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr

@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

@dataclass(frozen=True)
class Transpose(Expr):
    operand: Expr

@dataclass(frozen=True)
class Call(Expr):
    func: Expr
    args: List[Expr]

# ---- Indexing ----

@dataclass(frozen=True)
class IndexArg:
    line: int

@dataclass(frozen=True)
class Colon(IndexArg):
    pass

@dataclass(frozen=True)
class Range(IndexArg):
    start: Expr
    end: Expr

@dataclass(frozen=True)
class IndexExpr(IndexArg):
    expr: Expr

@dataclass(frozen=True)
class Index(Expr):
    base: Expr
    args: List[IndexArg]

# ---- Matrix literals ----

@dataclass(frozen=True)
class MatrixLit(Expr):
    rows: List[List[Expr]]

# ---- Statements / Program ----
@dataclass(frozen=True)
class Stmt:
    line: int

@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    expr: Expr

@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: Expr

@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: List[Stmt]
    else_body: List[Stmt]

@dataclass(frozen=True)
class While(Stmt):
    cond: Expr
    body: List[Stmt]

@dataclass(frozen=True)
class For(Stmt):
    var: str
    it: Expr # keep as Expr for now or change to Range later
    body: List[Stmt]

@dataclass(frozen=True)
class Program:
    body: List[Stmt]