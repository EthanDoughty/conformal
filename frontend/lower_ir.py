# Ethan Doughty
# lower_ir.py
"""Lower list-based syntax AST to typed IR AST."""

from __future__ import annotations
from typing import Any, List
from ir.ir import *


def lower_function(func: Any) -> FunctionDef:
    """Convert syntax function to IR FunctionDef.

    Args:
        func: ['function', line, output_vars, name, params, body]

    Returns:
        IR FunctionDef
    """
    tag, line, output_vars, name, params, body = func
    assert tag == "function"
    return FunctionDef(
        line=line,
        name=name,
        params=params,
        output_vars=output_vars,
        body=[lower_stmt(s) for s in body]
    )


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
        ast: List-based syntax AST with ['seq', item1, item2, ...]
        Items can be statements or function definitions.

    Returns:
        IR Program
    """
    assert isinstance(ast, list) and ast and ast[0] == "seq"
    body = []
    for item in ast[1:]:
        if isinstance(item, list) and item[0] == "function":
            body.append(lower_function(item))
        else:
            body.append(lower_stmt(item))
    return Program(body=body)


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

    if tag == "struct_assign":
        # ['struct_assign', line, base_name, fields, expr]
        line = stmt[1]
        base_name = stmt[2]
        fields = stmt[3]
        expr = lower_expr(stmt[4])
        return StructAssign(line=line, base_name=base_name, fields=fields, expr=expr)

    if tag == "cell_assign":
        # ['cell_assign', line, base_name, args, expr]
        line = stmt[1]
        base_name = stmt[2]
        args = [lower_index_arg(arg) for arg in stmt[3]]
        expr = lower_expr(stmt[4])
        return CellAssign(line=line, base_name=base_name, args=args, expr=expr)

    if tag == "index_assign":
        # ['index_assign', line, base_name, args, expr]
        line = stmt[1]
        base_name = stmt[2]
        args = [lower_index_arg(arg) for arg in stmt[3]]
        expr = lower_expr(stmt[4])
        return IndexAssign(line=line, base_name=base_name, args=args, expr=expr)

    if tag == "assign_multi":
        # ['assign_multi', line, targets, expr]
        line = stmt[1]
        targets = stmt[2]
        expr = lower_expr(stmt[3])
        return AssignMulti(line=line, targets=targets, expr=expr)

    if tag == "expr":
        expr = stmt[1]
        line = expr[1] if isinstance(expr, list) and len(expr) > 1 and isinstance(expr[1], int) else 0
        return ExprStmt(line=line, expr=lower_expr(expr))

    if tag == "if":
        # Updated for v0.11.0: handle 5-element format with elseifs
        # ['if', cond, then_body, elseifs, else_body]
        cond = lower_expr(stmt[1])
        then_body = [lower_stmt(x) for x in stmt[2]]
        elseifs = stmt[3]  # [[cond2, body2], ...]
        else_body = [lower_stmt(x) for x in stmt[4]]

        if not elseifs:
            # Simple if/else (backward compatible)
            return If(line=cond.line, cond=cond, then_body=then_body, else_body=else_body)
        else:
            # IfChain
            conditions = [cond] + [lower_expr(ec) for ec, _ in elseifs]
            bodies = [then_body] + [[lower_stmt(s) for s in body] for _, body in elseifs]
            return IfChain(line=cond.line, conditions=conditions, bodies=bodies, else_body=else_body)

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

    if tag == "return":
        return Return(line=stmt[1])

    if tag == "switch":
        # ['switch', expr, cases, otherwise_body]
        expr = lower_expr(stmt[1])
        cases = [(lower_expr(case_val), [lower_stmt(s) for s in case_body])
                 for case_val, case_body in stmt[2]]
        otherwise = [lower_stmt(s) for s in stmt[3]]
        return Switch(line=expr.line, expr=expr, cases=cases, otherwise=otherwise)

    if tag == "try":
        # ['try', try_body, catch_body]
        line = stmt[1][0][1] if stmt[1] and isinstance(stmt[1][0], list) and len(stmt[1][0]) > 1 else 0
        try_body = [lower_stmt(s) for s in stmt[1]]
        catch_body = [lower_stmt(s) for s in stmt[2]]
        return Try(line=line, try_body=try_body, catch_body=catch_body)

    if tag == "break":
        return Break(line=stmt[1])

    if tag == "continue":
        return Continue(line=stmt[1])

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

    if tag == "string":
        return StringLit(line=expr[1], value=expr[2])

    if tag == "neg":
        return Neg(line=expr[1], operand=lower_expr(expr[2]))

    if tag == "transpose":
        return Transpose(line=expr[1], operand=lower_expr(expr[2]))

    if tag == "field_access":
        # ['field_access', line, base_expr, field_name]
        return FieldAccess(line=expr[1], base=lower_expr(expr[2]), field=expr[3])

    if tag == "lambda":
        # ['lambda', line, params, body_expr]
        return Lambda(line=expr[1], params=expr[2], body=lower_expr(expr[3]))

    if tag == "func_handle":
        # ['func_handle', line, name]
        return FuncHandle(line=expr[1], name=expr[2])

    if tag == "end":
        # ['end', line]
        return End(line=expr[1])

    if tag == "apply":
        return Apply(line=expr[1], base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])

    if tag == "curly_apply":
        return CurlyApply(line=expr[1], base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])

    if tag == "matrix":
        return MatrixLit(line=expr[1], rows=[[lower_expr(elem) for elem in row] for row in expr[2]])

    if tag == "cell":
        return CellLit(line=expr[1], rows=[[lower_expr(elem) for elem in row] for row in expr[2]])

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