# Ethan Doughty
# analysis_ast.py
from __future__ import annotations
from typing import Any, List
from ir.ir import *

def lower_program(ast: Any) -> Program:
    assert isinstance(ast, list) and ast and ast[0] == "seq"
    return Program(body=[lower_stmt(s) for s in ast[1:]])

def lower_stmt(s: Any) -> Stmt:
    tag = s[0]

    if tag == "assign":
        line, name, expr = s[1], s[2], s[3]
        return Assign(line=line, name=name, expr=lower_expr(expr))

    if tag == "expr":
        expr = s[1]
        line = expr[1] if isinstance(expr, list) and len(expr) > 1 and isinstance(expr[1], int) else 0
        return ExprStmt(line=line, expr=lower_expr(expr))

    if tag == "if":
        cond = lower_expr(s[1])
        then_body = [lower_stmt(x) for x in s[2]]
        else_body = [lower_stmt(x) for x in s[3]]
        return If(line=cond.line, cond=cond, then_body=then_body, else_body=else_body)

    if tag == "while":
        cond = lower_expr(s[1])
        body = [lower_stmt(x) for x in s[2]]
        return While(line=cond.line, cond=cond, body=body)

    if tag == "for":
        # ['for', ['var', name], it_expr, body]
        var_node = s[1]
        var_name = var_node[1] if var_node[0] == "var" else var_node[2]
        it = lower_expr(s[2])
        body = [lower_stmt(x) for x in s[3]]
        return For(line=it.line, var=var_name, it=it, body=body)

    # fallback (better to raise later)
    return ExprStmt(line=0, expr=Const(line=0, value=0.0))

def lower_expr(e: Any) -> Expr:
    tag = e[0]

    if tag == "var":
        return Var(line=e[1], name=e[2])

    if tag == "const":
        return Const(line=e[1], value=float(e[2]))

    if tag == "neg":
        return Neg(line=e[1], operand=lower_expr(e[2]))

    if tag == "transpose":
        return Transpose(line=e[1], operand=lower_expr(e[2]))

    if tag == "call":
        return Call(line=e[1], func=lower_expr(e[2]), args=[lower_expr(a) for a in e[3]])

    if tag == "matrix":
        return MatrixLit(line=e[1], rows=[[lower_expr(x) for x in row] for row in e[2]])

    if tag == "index":
        base = lower_expr(e[2])
        args_list = e[3]
        lowered_args = [lower_index_arg(a) for a in args_list]
        return Index(line=e[1], base=base, args=lowered_args)

    if tag in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||", ":"}:
        return BinOp(line=e[1], op=tag, left=lower_expr(e[2]), right=lower_expr(e[3]))

    # fallback: unknown expression shape; keep something harmless
    return Const(line=0, value=0.0)

def lower_index_arg(a: Any) -> IndexArg:
    tag = a[0]

    if tag == "colon":
        return Colon(line=a[1])

    # Range inside indexing args is encoded as ":" in your current syntax AST
    if tag == ":":
        return Range(line=a[1], start=lower_expr(a[2]), end=lower_expr(a[3]))

    # If you ever emit ["range", ...] explicitly, support it too
    if tag == "range":
        return Range(line=a[1], start=lower_expr(a[2]), end=lower_expr(a[3]))

    # Otherwise it’s a normal expression subscript
    line = a[1] if isinstance(a, list) and len(a) > 1 and isinstance(a[1], int) else 0
    return IndexExpr(line=line, expr=lower_expr(a))