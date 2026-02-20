# Ethan Doughty
# ast_printer.py
"""Print the IR AST produced by parse_matlab."""
import sys
from dataclasses import fields
from typing import Any

from frontend.matlab_parser import parse_matlab
from ir.ir import (
    Program, Expr, Stmt, IndexArg,
    Var, Const, StringLit, BinOp, Neg, Not, Transpose,
    FieldAccess, Lambda, FuncHandle, End, Apply, CurlyApply,
    MatrixLit, CellLit, Colon, Range, IndexExpr,
    Assign, StructAssign, CellAssign, IndexAssign, IndexStructAssign,
    ExprStmt, If, IfChain, While, For, Switch, Try, Break, Continue,
    OpaqueStmt, FunctionDef, AssignMulti, Return,
)


def parse_file(path: str) -> Program:
    """Read a MATLAB file and return its IR Program."""
    with open(path, "r") as f:
        src = f.read()
    return parse_matlab(src)


def fmt_ir(node: Any, indent: int = 0) -> str:
    """Pretty-format an IR dataclass node."""
    pad = "  " * indent

    if node is None:
        return pad + "None"
    if isinstance(node, bool):
        return pad + repr(node)
    if isinstance(node, (int, float, str)):
        return pad + repr(node)
    if isinstance(node, list):
        if not node:
            return pad + "[]"
        lines = [pad + "["]
        for item in node:
            lines.append(fmt_ir(item, indent + 1) + ",")
        lines.append(pad + "]")
        return "\n".join(lines)
    if isinstance(node, tuple):
        if not node:
            return pad + "()"
        lines = [pad + "("]
        for item in node:
            lines.append(fmt_ir(item, indent + 1) + ",")
        lines.append(pad + ")")
        return "\n".join(lines)

    # IR dataclass nodes
    cls = type(node)
    cls_name = cls.__name__
    try:
        node_fields = fields(node)
    except TypeError:
        return pad + repr(node)

    if not node_fields:
        return pad + f"{cls_name}()"

    # One-line for simple leaf nodes
    if cls in (Var, Const, StringLit, End, Break, Continue, Return, Colon,
               FuncHandle, OpaqueStmt):
        field_parts = []
        for f in node_fields:
            val = getattr(node, f.name)
            field_parts.append(f"{f.name}={val!r}")
        return pad + f"{cls_name}({', '.join(field_parts)})"

    # Multi-line for complex nodes
    lines = [pad + f"{cls_name}("]
    for f in node_fields:
        val = getattr(node, f.name)
        formatted = fmt_ir(val, indent + 1)
        # If the value fits on one line, keep it inline
        if "\n" not in formatted:
            lines.append(pad + f"  {f.name}={formatted.strip()},")
        else:
            lines.append(pad + f"  {f.name}=")
            lines.append(formatted + ",")
    lines.append(pad + ")")
    return "\n".join(lines)


def usage() -> None:
    print("Usage: python3 ast_printer.py [--raw] <file.m>")
    print("  --raw     Print raw repr() of IR dataclasses")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    raw = False

    while args and args[0].startswith("--"):
        if args[0] == "--raw":
            raw = True
        else:
            usage()
        args = args[1:]

    if len(args) != 1:
        usage()

    path = args[0]
    try:
        ir = parse_file(path)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        sys.exit(1)

    print(f"==== IR for {path}")

    if raw:
        print(repr(ir))
    else:
        print(fmt_ir(ir))

    print(f"\n==== {len(ir.body)} top-level statements")


if __name__ == "__main__":
    main()
