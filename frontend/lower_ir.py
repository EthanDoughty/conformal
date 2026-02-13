# Ethan Doughty
# lower_ir.py
"""Lower list-based syntax AST to typed IR AST."""

from __future__ import annotations
from typing import Any, List
from ir.ir import *


def extract_targets_from_tokens(tokens: List[Any]) -> List[str]:
    """Conservatively extract target variable names from raw tokens.

    Supports:
    - IDENT = ...
    - IDENT(...) = ...
    - [A, B, ...] = ... (only ID/COMMA/NEWLINE/~ in brackets)

    Args:
        tokens: List of Token objects from recovered statement

    Returns:
        List of variable names that may be assigned
    """
    if not tokens:
        return []

    targets = []

    # Simple case: IDENT = ...
    if len(tokens) >= 2 and tokens[0].kind == "ID" and tokens[1].value == "=":
        return [tokens[0].value]

    # Function-style: IDENT(...) = ...
    if len(tokens) >= 3 and tokens[0].kind == "ID" and tokens[1].value == "(":
        # Find matching ), then check for =
        depth = 0
        for i, tok in enumerate(tokens):
            if tok.value == "(":
                depth += 1
            elif tok.value == ")":
                depth -= 1
                if depth == 0 and i + 1 < len(tokens) and tokens[i + 1].value == "=":
                    return [tokens[0].value]
                break

    # Destructuring: [A, B, ...] = ...
    # Enforce strict validation: only ID, COMMA, NEWLINE, or ~ inside brackets
    if len(tokens) >= 2 and tokens[0].value == "[":
        depth = 0
        bracket_end = -1
        for i, tok in enumerate(tokens):
            if tok.value == "[":
                depth += 1
            elif tok.value == "]":
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break

        if bracket_end > 0 and bracket_end + 1 < len(tokens) and tokens[bracket_end + 1].value == "=":
            # Validate bracket contents: only ID, COMMA, NEWLINE, ~
            valid_destructuring = True
            for j in range(1, bracket_end):
                tok = tokens[j]
                if tok.kind not in {"ID", "NEWLINE"} and tok.value not in {",", "~"}:
                    valid_destructuring = False
                    break

            if valid_destructuring:
                # Extract identifiers from inside brackets
                for j in range(1, bracket_end):
                    if tokens[j].kind == "ID":
                        targets.append(tokens[j].value)
                return targets

    return []


def lower_program(ast: Any) -> Program:
    """Convert syntax AST to IR Program.

    Args:
        ast: List-based syntax AST with ['seq', stmt1, stmt2, ...]

    Returns:
        IR Program
    """
    assert isinstance(ast, list) and ast and ast[0] == "seq"
    return Program(body=[lower_stmt(s) for s in ast[1:]])


def lower_stmt(stmt: Any) -> Stmt:
    """Convert syntax statement to IR Stmt.

    Args:
        stmt: List-based syntax statement

    Returns:
        IR Stmt
    """
    tag = stmt[0]

    if tag == "assign":
        line, name, expr = stmt[1], stmt[2], stmt[3]
        return Assign(line=line, name=name, expr=lower_expr(expr))

    if tag == "expr":
        expr = stmt[1]
        line = expr[1] if isinstance(expr, list) and len(expr) > 1 and isinstance(expr[1], int) else 0
        return ExprStmt(line=line, expr=lower_expr(expr))

    if tag == "if":
        cond = lower_expr(stmt[1])
        then_body = [lower_stmt(x) for x in stmt[2]]
        else_body = [lower_stmt(x) for x in stmt[3]]
        return If(line=cond.line, cond=cond, then_body=then_body, else_body=else_body)

    if tag == "while":
        cond = lower_expr(stmt[1])
        body = [lower_stmt(x) for x in stmt[2]]
        return While(line=cond.line, cond=cond, body=body)

    if tag == "for":
        # ['for', ['var', name], it_expr, body]
        var_node = stmt[1]
        var_name = var_node[1] if var_node[0] == "var" else var_node[2]
        iterator_expr = lower_expr(stmt[2])
        body = [lower_stmt(x) for x in stmt[3]]
        return For(line=iterator_expr.line, var=var_name, it=iterator_expr, body=body)

    if tag == "raw_stmt":
        # ['raw_stmt', line, tokens, raw_text]
        line = stmt[1]
        tokens = stmt[2]
        raw_text = stmt[3]
        targets = extract_targets_from_tokens(tokens)
        return OpaqueStmt(line=line, targets=targets, raw=raw_text)

    if tag == "skip":
        # Skip empty statements
        return ExprStmt(line=0, expr=Const(line=0, value=0.0))

    # Fallback for unexpected statement types
    return ExprStmt(line=0, expr=Const(line=0, value=0.0))


def lower_expr(expr: Any) -> Expr:
    """Convert syntax expression to IR Expr.

    Args:
        expr: List-based syntax expression

    Returns:
        IR Expr
    """
    tag = expr[0]

    if tag == "var":
        return Var(line=expr[1], name=expr[2])

    if tag == "const":
        return Const(line=expr[1], value=float(expr[2]))

    if tag == "neg":
        return Neg(line=expr[1], operand=lower_expr(expr[2]))

    if tag == "transpose":
        return Transpose(line=expr[1], operand=lower_expr(expr[2]))

    if tag == "call":
        return Call(line=expr[1], func=lower_expr(expr[2]), args=[lower_expr(arg) for arg in expr[3]])

    if tag == "apply":
        return Apply(line=expr[1], base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])

    if tag == "matrix":
        return MatrixLit(line=expr[1], rows=[[lower_expr(elem) for elem in row] for row in expr[2]])

    if tag == "index":
        base = lower_expr(expr[2])
        args_list = expr[3]
        lowered_args = [lower_index_arg(arg) for arg in args_list]
        return Index(line=expr[1], base=base, args=lowered_args)

    if tag in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||", ":"}:
        return BinOp(line=expr[1], op=tag, left=lower_expr(expr[2]), right=lower_expr(expr[3]))

    # Fallback for unexpected expression types
    return Const(line=0, value=0.0)


def lower_index_arg(index_arg: Any) -> IndexArg:
    """Convert syntax index argument to IR IndexArg.

    Args:
        index_arg: List-based syntax index argument

    Returns:
        IR IndexArg (Colon, Range, or IndexExpr)
    """
    tag = index_arg[0]

    if tag == "colon":
        return Colon(line=index_arg[1])

    # Range inside indexing args is encoded as ":" in the syntax AST
    if tag == ":":
        return Range(line=index_arg[1], start=lower_expr(index_arg[2]), end=lower_expr(index_arg[3]))

    # Also support explicit "range" tag if it exists
    if tag == "range":
        return Range(line=index_arg[1], start=lower_expr(index_arg[2]), end=lower_expr(index_arg[3]))

    # Otherwise it's a normal expression subscript
    line = index_arg[1] if isinstance(index_arg, list) and len(index_arg) > 1 and isinstance(index_arg[1], int) else 0
    return IndexExpr(line=line, expr=lower_expr(index_arg))