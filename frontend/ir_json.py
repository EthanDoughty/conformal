# Ethan Doughty
# ir_json.py
"""JSON serializer and deserializer for the ir.ir dataclass tree.

Usage:
    from frontend.ir_json import ir_to_json, ir_from_json
    json_str = ir_to_json(program)          # Program -> JSON string
    program   = ir_from_json(json_str)      # JSON string -> Program

CLI:
    python3 -m frontend.ir_json tests/basics/inner_dim_mismatch.m
"""

import json
import sys
import os
from typing import Any, Dict

from ir.ir import (
    Program,
    # Expr nodes
    Var, Const, StringLit, Neg, Not, BinOp, Transpose, FieldAccess,
    Lambda, FuncHandle, End, Apply, CurlyApply, MatrixLit, CellLit,
    # IndexArg nodes
    Colon, Range, IndexExpr,
    # Stmt nodes
    Assign, StructAssign, CellAssign, IndexAssign, IndexStructAssign,
    FieldIndexAssign, ExprStmt, If, While, For, IfChain, Switch, Try,
    Break, Continue, OpaqueStmt, FunctionDef, AssignMulti, Return,
)


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _ser_expr(node) -> Dict:
    """Dispatch to the correct expression serializer by class name."""
    return _EXPR_SERIALIZERS[type(node).__name__](node)


def _ser_index_arg(node) -> Dict:
    """Dispatch to the correct IndexArg serializer by class name."""
    return _INDEX_ARG_SERIALIZERS[type(node).__name__](node)


def _ser_stmt(node) -> Dict:
    """Dispatch to the correct statement serializer by class name."""
    return _STMT_SERIALIZERS[type(node).__name__](node)


def _ser_rows(rows):
    """Serialize a list-of-list-of-Expr (for MatrixLit / CellLit)."""
    return [[_ser_expr(e) for e in row] for row in rows]


def _ser_body(stmts):
    """Serialize a list of statements."""
    return [_ser_stmt(s) for s in stmts]


def _ser_args(args):
    """Serialize a list of IndexArg nodes."""
    return [_ser_index_arg(a) for a in args]


_EXPR_SERIALIZERS = {
    "Var": lambda n: {
        "type": "Var", "line": n.line, "col": n.col, "name": n.name
    },
    "Const": lambda n: {
        "type": "Const", "line": n.line, "col": n.col, "value": n.value
    },
    "StringLit": lambda n: {
        "type": "StringLit", "line": n.line, "col": n.col, "value": n.value
    },
    "Neg": lambda n: {
        "type": "Neg", "line": n.line, "col": n.col,
        "operand": _ser_expr(n.operand)
    },
    "Not": lambda n: {
        "type": "Not", "line": n.line, "col": n.col,
        "operand": _ser_expr(n.operand)
    },
    "BinOp": lambda n: {
        "type": "BinOp", "line": n.line, "col": n.col,
        "op": n.op,
        "left": _ser_expr(n.left),
        "right": _ser_expr(n.right),
    },
    "Transpose": lambda n: {
        "type": "Transpose", "line": n.line, "col": n.col,
        "operand": _ser_expr(n.operand)
    },
    "FieldAccess": lambda n: {
        "type": "FieldAccess", "line": n.line, "col": n.col,
        "base": _ser_expr(n.base),
        "field": n.field,
    },
    "Lambda": lambda n: {
        "type": "Lambda", "line": n.line, "col": n.col,
        "params": list(n.params),
        "body": _ser_expr(n.body),
    },
    "FuncHandle": lambda n: {
        "type": "FuncHandle", "line": n.line, "col": n.col, "name": n.name
    },
    "End": lambda n: {
        "type": "End", "line": n.line, "col": n.col
    },
    "Apply": lambda n: {
        "type": "Apply", "line": n.line, "col": n.col,
        "base": _ser_expr(n.base),
        "args": _ser_args(n.args),
    },
    "CurlyApply": lambda n: {
        "type": "CurlyApply", "line": n.line, "col": n.col,
        "base": _ser_expr(n.base),
        "args": _ser_args(n.args),
    },
    "MatrixLit": lambda n: {
        "type": "MatrixLit", "line": n.line, "col": n.col,
        "rows": _ser_rows(n.rows),
    },
    "CellLit": lambda n: {
        "type": "CellLit", "line": n.line, "col": n.col,
        "rows": _ser_rows(n.rows),
    },
}

_INDEX_ARG_SERIALIZERS = {
    "Colon": lambda n: {
        "type": "Colon", "line": n.line, "col": n.col
    },
    "Range": lambda n: {
        "type": "Range", "line": n.line, "col": n.col,
        "start": _ser_expr(n.start),
        "end": _ser_expr(n.end),
    },
    "IndexExpr": lambda n: {
        "type": "IndexExpr", "line": n.line, "col": n.col,
        "expr": _ser_expr(n.expr),
    },
}

_STMT_SERIALIZERS = {
    "Assign": lambda n: {
        "type": "Assign", "line": n.line, "col": n.col,
        "name": n.name,
        "expr": _ser_expr(n.expr),
    },
    "StructAssign": lambda n: {
        "type": "StructAssign", "line": n.line, "col": n.col,
        "base_name": n.base_name,
        "fields": list(n.fields),
        "expr": _ser_expr(n.expr),
    },
    "CellAssign": lambda n: {
        "type": "CellAssign", "line": n.line, "col": n.col,
        "base_name": n.base_name,
        "args": _ser_args(n.args),
        "expr": _ser_expr(n.expr),
    },
    "IndexAssign": lambda n: {
        "type": "IndexAssign", "line": n.line, "col": n.col,
        "base_name": n.base_name,
        "args": _ser_args(n.args),
        "expr": _ser_expr(n.expr),
    },
    "IndexStructAssign": lambda n: {
        "type": "IndexStructAssign", "line": n.line, "col": n.col,
        "base_name": n.base_name,
        "index_args": _ser_args(n.index_args),
        "index_kind": n.index_kind,
        "fields": list(n.fields),
        "expr": _ser_expr(n.expr),
    },
    "FieldIndexAssign": lambda n: {
        "type": "FieldIndexAssign", "line": n.line, "col": n.col,
        "base_name": n.base_name,
        "prefix_fields": list(n.prefix_fields),
        "index_args": _ser_args(n.index_args),
        "index_kind": n.index_kind,
        "suffix_fields": list(n.suffix_fields),
        "expr": _ser_expr(n.expr),
    },
    "ExprStmt": lambda n: {
        "type": "ExprStmt", "line": n.line, "col": n.col,
        "expr": _ser_expr(n.expr),
    },
    "If": lambda n: {
        "type": "If", "line": n.line, "col": n.col,
        "cond": _ser_expr(n.cond),
        "then_body": _ser_body(n.then_body),
        "else_body": _ser_body(n.else_body),
    },
    "While": lambda n: {
        "type": "While", "line": n.line, "col": n.col,
        "cond": _ser_expr(n.cond),
        "body": _ser_body(n.body),
    },
    "For": lambda n: {
        "type": "For", "line": n.line, "col": n.col,
        "var": n.var,
        "it": _ser_expr(n.it),
        "body": _ser_body(n.body),
    },
    "IfChain": lambda n: {
        "type": "IfChain", "line": n.line, "col": n.col,
        "conditions": [_ser_expr(c) for c in n.conditions],
        "bodies": [_ser_body(b) for b in n.bodies],
        "else_body": _ser_body(n.else_body),
    },
    "Switch": lambda n: {
        "type": "Switch", "line": n.line, "col": n.col,
        "expr": _ser_expr(n.expr),
        "cases": [{"value": _ser_expr(v), "body": _ser_body(b)} for v, b in n.cases],
        "otherwise": _ser_body(n.otherwise),
    },
    "Try": lambda n: {
        "type": "Try", "line": n.line, "col": n.col,
        "try_body": _ser_body(n.try_body),
        "catch_body": _ser_body(n.catch_body),
    },
    "Break": lambda n: {
        "type": "Break", "line": n.line, "col": n.col
    },
    "Continue": lambda n: {
        "type": "Continue", "line": n.line, "col": n.col
    },
    "Return": lambda n: {
        "type": "Return", "line": n.line, "col": n.col
    },
    "OpaqueStmt": lambda n: {
        "type": "OpaqueStmt", "line": n.line, "col": n.col,
        "targets": list(n.targets),
        "raw": n.raw,
    },
    "FunctionDef": lambda n: {
        "type": "FunctionDef", "line": n.line, "col": n.col,
        "name": n.name,
        "params": list(n.params),
        "output_vars": list(n.output_vars),
        "body": _ser_body(n.body),
    },
    "AssignMulti": lambda n: {
        "type": "AssignMulti", "line": n.line, "col": n.col,
        "targets": list(n.targets),
        "expr": _ser_expr(n.expr),
    },
}


def _ser_program(program: Program) -> Dict:
    return {
        "type": "Program",
        "body": _ser_body(program.body),
    }


def ir_to_json(program: Program) -> str:
    """Serialize ir.ir.Program to a JSON string.

    Args:
        program: The Program IR node to serialize.

    Returns:
        Pretty-printed JSON string with 2-space indentation.
    """
    return json.dumps(_ser_program(program), indent=2)


# ---------------------------------------------------------------------------
# Deserializer
# ---------------------------------------------------------------------------

def _build_expr(d: Dict):
    """Reconstruct an Expr dataclass from a dict."""
    return _EXPR_BUILDERS[d["type"]](d)


def _build_index_arg(d: Dict):
    """Reconstruct an IndexArg dataclass from a dict."""
    return _INDEX_ARG_BUILDERS[d["type"]](d)


def _build_stmt(d: Dict):
    """Reconstruct a Stmt dataclass from a dict."""
    return _STMT_BUILDERS[d["type"]](d)


def _build_body(lst):
    """Reconstruct a list of Stmt nodes."""
    return [_build_stmt(s) for s in lst]


def _build_args(lst):
    """Reconstruct a list of IndexArg nodes."""
    return [_build_index_arg(a) for a in lst]


def _build_rows(lst):
    """Reconstruct a list-of-list-of-Expr (for MatrixLit / CellLit)."""
    return [[_build_expr(e) for e in row] for row in lst]


_EXPR_BUILDERS = {
    "Var": lambda d: Var(
        line=d["line"], name=d["name"], col=d.get("col", 0)
    ),
    "Const": lambda d: Const(
        line=d["line"], value=d["value"], col=d.get("col", 0)
    ),
    "StringLit": lambda d: StringLit(
        line=d["line"], value=d["value"], col=d.get("col", 0)
    ),
    "Neg": lambda d: Neg(
        line=d["line"], operand=_build_expr(d["operand"]), col=d.get("col", 0)
    ),
    "Not": lambda d: Not(
        line=d["line"], operand=_build_expr(d["operand"]), col=d.get("col", 0)
    ),
    "BinOp": lambda d: BinOp(
        line=d["line"],
        op=d["op"],
        left=_build_expr(d["left"]),
        right=_build_expr(d["right"]),
        col=d.get("col", 0),
    ),
    "Transpose": lambda d: Transpose(
        line=d["line"], operand=_build_expr(d["operand"]), col=d.get("col", 0)
    ),
    "FieldAccess": lambda d: FieldAccess(
        line=d["line"],
        base=_build_expr(d["base"]),
        field=d["field"],
        col=d.get("col", 0),
    ),
    "Lambda": lambda d: Lambda(
        line=d["line"],
        params=d["params"],
        body=_build_expr(d["body"]),
        col=d.get("col", 0),
    ),
    "FuncHandle": lambda d: FuncHandle(
        line=d["line"], name=d["name"], col=d.get("col", 0)
    ),
    "End": lambda d: End(
        line=d["line"], col=d.get("col", 0)
    ),
    "Apply": lambda d: Apply(
        line=d["line"],
        base=_build_expr(d["base"]),
        args=_build_args(d["args"]),
        col=d.get("col", 0),
    ),
    "CurlyApply": lambda d: CurlyApply(
        line=d["line"],
        base=_build_expr(d["base"]),
        args=_build_args(d["args"]),
        col=d.get("col", 0),
    ),
    "MatrixLit": lambda d: MatrixLit(
        line=d["line"],
        rows=_build_rows(d["rows"]),
        col=d.get("col", 0),
    ),
    "CellLit": lambda d: CellLit(
        line=d["line"],
        rows=_build_rows(d["rows"]),
        col=d.get("col", 0),
    ),
}

_INDEX_ARG_BUILDERS = {
    "Colon": lambda d: Colon(
        line=d["line"], col=d.get("col", 0)
    ),
    "Range": lambda d: Range(
        line=d["line"],
        start=_build_expr(d["start"]),
        end=_build_expr(d["end"]),
        col=d.get("col", 0),
    ),
    "IndexExpr": lambda d: IndexExpr(
        line=d["line"],
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
}

_STMT_BUILDERS = {
    "Assign": lambda d: Assign(
        line=d["line"], name=d["name"],
        expr=_build_expr(d["expr"]), col=d.get("col", 0)
    ),
    "StructAssign": lambda d: StructAssign(
        line=d["line"],
        base_name=d["base_name"],
        fields=d["fields"],
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
    "CellAssign": lambda d: CellAssign(
        line=d["line"],
        base_name=d["base_name"],
        args=_build_args(d["args"]),
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
    "IndexAssign": lambda d: IndexAssign(
        line=d["line"],
        base_name=d["base_name"],
        args=_build_args(d["args"]),
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
    "IndexStructAssign": lambda d: IndexStructAssign(
        line=d["line"],
        base_name=d["base_name"],
        index_args=_build_args(d["index_args"]),
        index_kind=d["index_kind"],
        fields=d["fields"],
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
    "FieldIndexAssign": lambda d: FieldIndexAssign(
        line=d["line"],
        base_name=d["base_name"],
        prefix_fields=d["prefix_fields"],
        index_args=_build_args(d["index_args"]),
        index_kind=d["index_kind"],
        suffix_fields=d["suffix_fields"],
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
    "ExprStmt": lambda d: ExprStmt(
        line=d["line"], expr=_build_expr(d["expr"]), col=d.get("col", 0)
    ),
    "If": lambda d: If(
        line=d["line"],
        cond=_build_expr(d["cond"]),
        then_body=_build_body(d["then_body"]),
        else_body=_build_body(d["else_body"]),
        col=d.get("col", 0),
    ),
    "While": lambda d: While(
        line=d["line"],
        cond=_build_expr(d["cond"]),
        body=_build_body(d["body"]),
        col=d.get("col", 0),
    ),
    "For": lambda d: For(
        line=d["line"],
        var=d["var"],
        it=_build_expr(d["it"]),
        body=_build_body(d["body"]),
        col=d.get("col", 0),
    ),
    "IfChain": lambda d: IfChain(
        line=d["line"],
        conditions=[_build_expr(c) for c in d["conditions"]],
        bodies=[_build_body(b) for b in d["bodies"]],
        else_body=_build_body(d["else_body"]),
        col=d.get("col", 0),
    ),
    "Switch": lambda d: Switch(
        line=d["line"],
        expr=_build_expr(d["expr"]),
        cases=[(_build_expr(c["value"]), _build_body(c["body"])) for c in d["cases"]],
        otherwise=_build_body(d["otherwise"]),
        col=d.get("col", 0),
    ),
    "Try": lambda d: Try(
        line=d["line"],
        try_body=_build_body(d["try_body"]),
        catch_body=_build_body(d["catch_body"]),
        col=d.get("col", 0),
    ),
    "Break": lambda d: Break(line=d["line"], col=d.get("col", 0)),
    "Continue": lambda d: Continue(line=d["line"], col=d.get("col", 0)),
    "Return": lambda d: Return(line=d["line"], col=d.get("col", 0)),
    "OpaqueStmt": lambda d: OpaqueStmt(
        line=d["line"],
        targets=d["targets"],
        raw=d.get("raw", ""),
        col=d.get("col", 0),
    ),
    "FunctionDef": lambda d: FunctionDef(
        line=d["line"],
        name=d["name"],
        params=d["params"],
        output_vars=d["output_vars"],
        body=_build_body(d["body"]),
        col=d.get("col", 0),
    ),
    "AssignMulti": lambda d: AssignMulti(
        line=d["line"],
        targets=d["targets"],
        expr=_build_expr(d["expr"]),
        col=d.get("col", 0),
    ),
}


def ir_from_json(json_str: str) -> Program:
    """Deserialize a JSON string to an ir.ir.Program.

    Args:
        json_str: JSON string produced by ir_to_json or the F# conformal-parse binary.

    Returns:
        Program dataclass instance reconstructed from the JSON.
    """
    data = json.loads(json_str)
    assert data["type"] == "Program", f"Expected 'Program', got '{data['type']}'"
    return Program(body=_build_body(data["body"]))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_golden(input_path: str, output_path: str) -> None:
    """Write the JSON for a single .m file to output_path.

    Used for generating golden files to debug F# parser divergences.

    Args:
        input_path: Path to a MATLAB source file (.m).
        output_path: Path where the JSON will be written.
    """
    from frontend.matlab_parser import parse_matlab
    src = open(input_path, encoding="utf-8", errors="replace").read()
    program = parse_matlab(src)
    json_str = ir_to_json(program)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m frontend.ir_json <file.m>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    # Allow running from any directory by resolving relative to cwd
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)

    # Ensure the repo root is on sys.path so ir and frontend are importable
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from frontend.matlab_parser import parse_matlab
    src = open(file_path, encoding="utf-8", errors="replace").read()
    program = parse_matlab(src)
    print(ir_to_json(program))
