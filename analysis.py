# Ethan Doughty
# analysis.py

from __future__ import annotations
from typing import Any, List, Tuple, Union

from matlab_parser import parse_matlab
from env import Env
from shapes import (
    Shape,
    shape_of_zeros,
    shape_of_ones,
    shape_of_colon,
    dims_definitely_conflict,
    join_dim,
)

Dim = Union[int, str, None]

def expr_to_dim(
        expr: Any, 
        env: Env
    ) -> Dim:
    """Try to interpret an expression as a dimension (int or symbolic). For the tests, dims are either numeric constants or scalars"""
    tag = expr[0]

    if tag == "const":
        val = expr[2]
        if float(val).is_integer():
            return int(val)
        return None

    if tag == "var":
        name = expr[2]
        return name

    # Anything more complex: don't know
    return None

def pretty_expr(expr):
    tag = expr[0]

    if tag == "var":
        return expr[2]
    if tag == "const":
        return str(expr[2])
    if tag in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||"}:
        return f"({pretty_expr(expr[2])} {tag} {pretty_expr(expr[3])})"
    if tag == "transpose":
        return pretty_expr(expr[2]) + "'"
    if tag == "call":
        func = pretty_expr(expr[2])
        args = ", ".join(pretty_expr(a) for a in expr[3])
        return f"{func}({args})"
    if tag == "index":
        base = pretty_expr(expr[2])
        idx = pretty_expr(expr[3])
        return f"{base}({idx})"
    if tag == "neg":
        return f"(-{pretty_expr(expr[2])})"
    return tag


# Expression analysis
def eval_expr(
        expr: Any, 
        env: Env, 
        warnings: List[str]
    ) -> Shape:
    tag = expr[0]

    if tag == "var":
        name = expr[2]
        return env.get(name)

    if tag == "const":
        return Shape.scalar()

    if tag == "call":
        # ['call', line, func_expr, args]
        func_expr = expr[2]
        args = expr[3]

        # Only handle calls where function is a simple variable name
        if func_expr[0] == "var":
            fname = func_expr[2]

            if fname == "zeros" or fname == "ones":
                if len(args) == 2:
                    r_dim = expr_to_dim(args[0], env)
                    c_dim = expr_to_dim(args[1], env)
                    if fname == "zeros":
                        return shape_of_zeros(r_dim, c_dim)
                    else:
                        return shape_of_ones(r_dim, c_dim)

        # Unknown function or unsupported
        return Shape.scalar()

    if tag == "transpose":
        inner = eval_expr(expr[2], env, warnings)
        if inner.is_matrix():
            return Shape.matrix(inner.cols, inner.rows)
        return inner

    if tag == "index":
        base_shape = eval_expr(expr[2], env, warnings)
        # For now I'm treating the indexing result as a scalar or unknown
        if base_shape.is_matrix():
            return Shape.scalar()
        return Shape.unknown()
    
    if tag == "neg":
        inner = eval_expr(expr[2], env, warnings)
        return inner

    op = tag
    if op in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||", ":"}:
        line = expr[1]
        left_expr = expr[2]
        right_expr = expr[3]
        left_shape = eval_expr(left_expr, env, warnings)
        right_shape = eval_expr(right_expr, env, warnings)
        return eval_binop(op, left_shape, right_shape, warnings, left_expr, right_expr, line)

    # Fallback
    return Shape.unknown()


def eval_binop(
        op: str, 
        left: Shape, 
        right: Shape, 
        warnings: List[str], 
        left_expr: Any, 
        right_expr: Any,
        line: int
    ) -> Shape:
    """Evaluate binary operator shapes and emit dimension mismatch warnings where we can prove incompatibility"""

    if op in {"==", "~=", "<", "<=", ">", ">="}:
        # Warn if someone compares matrices (MATLAB comparisons are elementwise and can produce a logical matrix)
        if (left.is_matrix() and right.is_scalar()) or (left.is_scalar() and right.is_matrix()):
            warnings.append(
                f"Line {line}: Suspicious comparison between matrix and scalar in "
                f"{pretty_expr([op, line, left_expr, right_expr])} ({left} vs {right}). "
                f"In MATLAB this is elementwise and may produce a logical matrix."
            )
        elif left.is_matrix() and right.is_matrix():
            warnings.append(
                f"Line {line}: Matrix-to-matrix comparison in "
                f"{pretty_expr([op, line, left_expr, right_expr])} ({left} vs {right}). "
                f"In MATLAB this is elementwise and may produce a logical matrix."
            )
        return Shape.scalar()

    if op in {"&&", "||"}:
        # also warn if logical ops are applied to non-scalars
        if left.is_matrix() or right.is_matrix():
            warnings.append(
                f"Line {line}: Logical operator {op} used with non-scalar operand(s) in "
                f"{pretty_expr([op, line, left_expr, right_expr])} ({left} vs {right})."
            )
        return Shape.scalar()

    # Colon: 1:n style vector
    if op == ":":
        return Shape.matrix(1, None)

    # Scalar expansion: if one is scalar, return the other shape, no mismatch
    if left.is_scalar() and not right.is_scalar():
        return right
    if right.is_scalar() and not left.is_scalar():
        return left

    # Elementwise requires the same shape
    if op in {"+", "-", ".*", "./", "/"}:
        return elementwise_shape(op, left, right, warnings, left_expr, right_expr, line)

    # Matrix multiply
    if op == "*":
        return matmul_shape(left, right, warnings, left_expr, right_expr, line)

    # Fallback
    return Shape.unknown()


def elementwise_shape(
        op: str, 
        left: Shape, 
        right: Shape, 
        warnings: List[str], 
        left_expr: Any, 
        right_expr: Any,
        line: int
    ) -> Shape:
    """Elementwise operations require matching shapes when both operands are matrices."""
    # Unknowns -> unknown
    if left.is_unknown() or right.is_unknown():
        return Shape.unknown()

    # Both scalars
    if left.is_scalar() and right.is_scalar():
        return Shape.scalar()

    # Both matrices
    if left.is_matrix() and right.is_matrix():
        r_conflict = dims_definitely_conflict(left.rows, right.rows)
        c_conflict = dims_definitely_conflict(left.cols, right.cols)
        if r_conflict or c_conflict:
            warnings.append(
                f"Line {line}: Elementwise {op} mismatch in "
                f"{pretty_expr([op, line, left_expr, right_expr])}: {left} vs {right}"
)
        # Result has joined dimensions
        rows = join_dim(left.rows, right.rows)
        cols = join_dim(left.cols, right.cols)
        return Shape.matrix(rows, cols)

    # Anything else is unknown
    return Shape.unknown()


def matmul_shape(
        left: Shape, 
        right: Shape, 
        warnings: List[str], 
        left_expr: Any, 
        right_expr: Any,
        line: int
    ) -> Shape:
    """Matrix multiplication: A * B"""
    # Scalar * scalar
    if left.is_scalar() and right.is_scalar():
        return Shape.scalar()

    # Scalar * matrix or matrix * scalar 
    if left.is_scalar() and right.is_matrix():
        return right
    if right.is_scalar() and left.is_matrix():
        return left

    # Both matrices, inner dimension must match
    if left.is_matrix() and right.is_matrix():
        inner_left = left.cols
        inner_right = right.rows
        if dims_definitely_conflict(inner_left, inner_right):
            warnings.append(
                f"Line {line}: Dimension mismatch in expression "
                f"{pretty_expr(['*', line, left_expr, right_expr])}: "
                f"inner dims {left.cols} vs {right.rows} (shapes {left} and {right})"
            )
        # Result shape is rows_left x cols_right, even if inner dims are unknown
        return Shape.matrix(left.rows, right.cols)

    # Anything else is unknown
    return Shape.unknown()


def analyze_stmt(
        stmt: Any, 
        env: Env, 
        warnings: List[str]
    ) -> Env:
    tag = stmt[0]

    if tag == "assign":
        name = stmt[2]
        expr = stmt[3]
        shape = eval_expr(expr, env, warnings)
        env.set(name, shape)
        return env

    if tag == "expr":
        _ = eval_expr(stmt[1], env, warnings)
        return env

    if tag == "skip":
        return env

    if tag == "for":
        body = stmt[3]
        # naive: analyze body once
        for s in body:
            analyze_stmt(s, env, warnings)
        return env

    if tag == "while":
        
        cond = stmt[1]
        body = stmt[2]

        # analyze the condition so comparison/logical warnings trigger
        _ = eval_expr(cond, env, warnings)

        for s in body:
            analyze_stmt(s, env, warnings)
        return env

    if tag == "if":

        cond = stmt[1]
        then_body = stmt[2]
        else_body = stmt[3]

        _ = eval_expr(cond, env, warnings)

        then_env = env.copy()
        else_env = env.copy()
        for s in then_body:
            analyze_stmt(s, then_env, warnings)
        for s in else_body:
            analyze_stmt(s, else_env, warnings)

        # merge environments
        from env import join_env
        merged = join_env(then_env, else_env)
        env.bindings = merged.bindings  # update in place
        return env

    # Unknown statement
    return env


def analyze_program(ast_root: Any) -> Tuple[Env, List[str]]:
    """Given a parsed AST, run the shape analysis and return (final_env, warnings)"""
    assert ast_root[0] == "seq"
    env = Env()
    warnings: List[str] = []

    for stmt in ast_root[1:]:
        analyze_stmt(stmt, env, warnings)

    return env, warnings