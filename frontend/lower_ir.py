# Ethan Doughty
# lower_ir.py
"""Lower list-based syntax AST to typed IR AST."""

from __future__ import annotations
import sys
from typing import Any, List
from ir.ir import *


def _unpack_lc(lc: Any):
    """Unpack a (line, col) tuple from a syntax AST position field.

    Returns (line, col) integers. Accepts a tuple (from the updated parser) or
    a plain int (legacy path, col defaults to 0).
    """
    if isinstance(lc, tuple):
        return lc[0], lc[1]
    if isinstance(lc, int):
        return lc, 0
    return 0, 0


def lower_function(func: Any) -> FunctionDef:
    """Convert syntax function to IR FunctionDef.

    Args:
        func: ['function', (line, col), output_vars, name, params, body]

    Returns:
        IR FunctionDef
    """
    tag, lc, output_vars, name, params, body = func
    assert tag == "function"
    line, col = _unpack_lc(lc)
    return FunctionDef(
        line=line,
        col=col,
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
        line, col = _unpack_lc(stmt[1])
        name, expr = stmt[2], stmt[3]
        return Assign(line=line, col=col, name=name, expr=lower_expr(expr))

    if tag == "struct_assign":
        # ['struct_assign', (line, col), base_name, fields, expr]
        line, col = _unpack_lc(stmt[1])
        base_name = stmt[2]
        fields = stmt[3]
        expr = lower_expr(stmt[4])
        return StructAssign(line=line, col=col, base_name=base_name, fields=fields, expr=expr)

    if tag == "cell_assign":
        # ['cell_assign', (line, col), base_name, args, expr]
        line, col = _unpack_lc(stmt[1])
        base_name = stmt[2]
        args = [lower_index_arg(arg) for arg in stmt[3]]
        expr = lower_expr(stmt[4])
        return CellAssign(line=line, col=col, base_name=base_name, args=args, expr=expr)

    if tag == "index_assign":
        # ['index_assign', (line, col), base_name, args, expr]
        line, col = _unpack_lc(stmt[1])
        base_name = stmt[2]
        args = [lower_index_arg(arg) for arg in stmt[3]]
        expr = lower_expr(stmt[4])
        return IndexAssign(line=line, col=col, base_name=base_name, args=args, expr=expr)

    if tag == "index_struct_assign":
        # ['index_struct_assign', (line, col), base_name, index_args, index_kind, fields, expr]
        line, col = _unpack_lc(stmt[1])
        base_name = stmt[2]
        index_args = [lower_index_arg(arg) for arg in stmt[3]]
        index_kind = stmt[4]
        fields = stmt[5]
        expr = lower_expr(stmt[6])
        return IndexStructAssign(line=line, col=col, base_name=base_name, index_args=index_args,
                                 index_kind=index_kind, fields=fields, expr=expr)

    if tag == "assign_multi":
        # ['assign_multi', (line, col), targets, expr]
        line, col = _unpack_lc(stmt[1])
        targets = stmt[2]
        expr = lower_expr(stmt[3])
        return AssignMulti(line=line, col=col, targets=targets, expr=expr)

    if tag == "expr":
        expr = stmt[1]
        if isinstance(expr, list) and len(expr) > 1 and isinstance(expr[1], (int, tuple)):
            line, col = _unpack_lc(expr[1])
        else:
            line, col = 0, 0
        return ExprStmt(line=line, col=col, expr=lower_expr(expr))

    if tag == "if":
        # Updated for v0.11.0: handle 5-element format with elseifs
        # ['if', cond, then_body, elseifs, else_body]
        cond = lower_expr(stmt[1])
        then_body = [lower_stmt(x) for x in stmt[2]]
        elseifs = stmt[3]  # [[cond2, body2], ...]
        else_body = [lower_stmt(x) for x in stmt[4]]

        if not elseifs:
            # Simple if/else (backward compatible)
            return If(line=cond.line, col=cond.col, cond=cond, then_body=then_body, else_body=else_body)
        else:
            # IfChain
            conditions = [cond] + [lower_expr(ec) for ec, _ in elseifs]
            bodies = [then_body] + [[lower_stmt(s) for s in body] for _, body in elseifs]
            return IfChain(line=cond.line, col=cond.col, conditions=conditions, bodies=bodies, else_body=else_body)

    if tag == "while":
        cond = lower_expr(stmt[1])
        body = [lower_stmt(x) for x in stmt[2]]
        return While(line=cond.line, col=cond.col, cond=cond, body=body)

    if tag == "for":
        # ['for', ['var', name], it_expr, body]
        var_node = stmt[1]
        var_name = var_node[1] if var_node[0] == "var" else var_node[2]
        iterator_expr = lower_expr(stmt[2])
        body = [lower_stmt(x) for x in stmt[3]]
        return For(line=iterator_expr.line, col=iterator_expr.col, var=var_name, it=iterator_expr, body=body)

    if tag == "return":
        line, col = _unpack_lc(stmt[1])
        return Return(line=line, col=col)

    if tag == "switch":
        # ['switch', expr, cases, otherwise_body]
        expr = lower_expr(stmt[1])
        cases = [(lower_expr(case_val), [lower_stmt(s) for s in case_body])
                 for case_val, case_body in stmt[2]]
        otherwise = [lower_stmt(s) for s in stmt[3]]
        return Switch(line=expr.line, col=expr.col, expr=expr, cases=cases, otherwise=otherwise)

    if tag == "try":
        # ['try', try_body, catch_body]
        first = stmt[1][0] if stmt[1] else None
        if first and isinstance(first, list) and len(first) > 1:
            line, col = _unpack_lc(first[1])
        else:
            line, col = 0, 0
        try_body = [lower_stmt(s) for s in stmt[1]]
        catch_body = [lower_stmt(s) for s in stmt[2]]
        return Try(line=line, col=col, try_body=try_body, catch_body=catch_body)

    if tag == "break":
        line, col = _unpack_lc(stmt[1])
        return Break(line=line, col=col)

    if tag == "continue":
        line, col = _unpack_lc(stmt[1])
        return Continue(line=line, col=col)

    if tag == "global_decl":
        # ['global_decl', (line, col), var_names]
        line, col = _unpack_lc(stmt[1])
        var_names = stmt[2]
        raw_text = "global " + " ".join(var_names)
        return OpaqueStmt(line=line, col=col, targets=var_names, raw=raw_text)

    if tag == "raw_stmt":
        # ['raw_stmt', (line, col), tokens, raw_text]
        line, col = _unpack_lc(stmt[1])
        tokens = stmt[2]
        raw_text = stmt[3]
        targets = extract_targets_from_tokens(tokens)
        return OpaqueStmt(line=line, col=col, targets=targets, raw=raw_text)

    if tag == "skip":
        # Intentional no-op: parser emits 'skip' for continuation lines and empty statements.
        # Returning a harmless ExprStmt(Const(0.0)) here is correct; it produces no warnings
        # and is filtered out during analysis like any other side-effect-free expression.
        return ExprStmt(line=0, expr=Const(line=0, value=0.0))

    # Fallback for unrecognized statement tags: use OpaqueStmt so the analyzer
    # havoces any written targets and emits W_UNSUPPORTED_STMT, which is the
    # correct conservative recovery path for unknown statement forms.
    if len(stmt) > 1 and isinstance(stmt[1], (int, tuple)):
        tag_line, tag_col = _unpack_lc(stmt[1])
    else:
        tag_line, tag_col = 0, 0
    print(f"Warning: lower_stmt: unrecognized tag {tag!r}", file=sys.stderr)
    return OpaqueStmt(line=tag_line, col=tag_col, targets=[], raw=f"<unrecognized: {tag}>")


def lower_expr(expr: Any) -> Expr:
    """Convert syntax expression to IR Expr.

    Args:
        expr: List-based syntax expression

    Returns:
        IR Expr
    """
    tag = expr[0]

    if tag == "var":
        line, col = _unpack_lc(expr[1])
        return Var(line=line, col=col, name=expr[2])

    if tag == "const":
        line, col = _unpack_lc(expr[1])
        return Const(line=line, col=col, value=float(expr[2]))

    if tag == "string":
        line, col = _unpack_lc(expr[1])
        return StringLit(line=line, col=col, value=expr[2])

    if tag == "neg":
        line, col = _unpack_lc(expr[1])
        return Neg(line=line, col=col, operand=lower_expr(expr[2]))

    if tag == "not":
        line, col = _unpack_lc(expr[1])
        return Not(line=line, col=col, operand=lower_expr(expr[2]))

    if tag == "transpose":
        line, col = _unpack_lc(expr[1])
        return Transpose(line=line, col=col, operand=lower_expr(expr[2]))

    if tag == "field_access":
        # ['field_access', (line, col), base_expr, field_name]
        line, col = _unpack_lc(expr[1])
        return FieldAccess(line=line, col=col, base=lower_expr(expr[2]), field=expr[3])

    if tag == "lambda":
        # ['lambda', (line, col), params, body_expr]
        line, col = _unpack_lc(expr[1])
        return Lambda(line=line, col=col, params=expr[2], body=lower_expr(expr[3]))

    if tag == "func_handle":
        # ['func_handle', (line, col), name]
        line, col = _unpack_lc(expr[1])
        return FuncHandle(line=line, col=col, name=expr[2])

    if tag == "end":
        # ['end', (line, col)]
        line, col = _unpack_lc(expr[1])
        return End(line=line, col=col)

    if tag == "apply":
        line, col = _unpack_lc(expr[1])
        return Apply(line=line, col=col, base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])

    if tag == "curly_apply":
        line, col = _unpack_lc(expr[1])
        return CurlyApply(line=line, col=col, base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])

    if tag == "matrix":
        line, col = _unpack_lc(expr[1])
        return MatrixLit(line=line, col=col, rows=[[lower_expr(elem) for elem in row] for row in expr[2]])

    if tag == "cell":
        line, col = _unpack_lc(expr[1])
        return CellLit(line=line, col=col, rows=[[lower_expr(elem) for elem in row] for row in expr[2]])

    if tag in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||", ":", "^", ".^", "\\", "&", "|"}:
        line, col = _unpack_lc(expr[1])
        return BinOp(line=line, col=col, op=tag, left=lower_expr(expr[2]), right=lower_expr(expr[3]))

    # Unrecognized expression tags cannot be soundly replaced with any constant.
    # Raise immediately so the bug surfaces in tests rather than silently
    # injecting a 0.0 that changes program semantics.
    raise ValueError(f"lower_expr: unrecognized tag {tag!r}")


def lower_index_arg(index_arg: Any) -> IndexArg:
    """Convert syntax index argument to IR IndexArg.

    Args:
        index_arg: List-based syntax index argument

    Returns:
        IR IndexArg (Colon, Range, or IndexExpr)
    """
    tag = index_arg[0]

    if tag == "colon":
        line, col = _unpack_lc(index_arg[1])
        return Colon(line=line, col=col)

    # Range inside indexing args is encoded as ":" in the syntax AST
    if tag == ":":
        line, col = _unpack_lc(index_arg[1])
        return Range(line=line, col=col, start=lower_expr(index_arg[2]), end=lower_expr(index_arg[3]))

    # Also support explicit "range" tag if it exists
    if tag == "range":
        line, col = _unpack_lc(index_arg[1])
        return Range(line=line, col=col, start=lower_expr(index_arg[2]), end=lower_expr(index_arg[3]))

    # Otherwise it's a normal expression subscript
    if isinstance(index_arg, list) and len(index_arg) > 1 and isinstance(index_arg[1], (int, tuple)):
        line, col = _unpack_lc(index_arg[1])
    else:
        line, col = 0, 0
    return IndexExpr(line=line, col=col, expr=lower_expr(index_arg))