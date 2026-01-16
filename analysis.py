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
    add_dim,
    sum_dims,
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
    if tag == "colon":
        return ":"
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
        args = expr[3]
        args_s = ", ".join(pretty_expr(a) for a in args)
        return f"{base}({args_s})"
    if tag == "neg":
        return f"(-{pretty_expr(expr[2])})"
    return tag

def shapes_definitely_incompatible(old: Shape, new: Shape) -> bool:
    # If either is unknown, don't claim incompatibility
    if old.is_unknown() or new.is_unknown():
        return False

    # Scalar vs matrix is definitely incompatible for reassignment
    if old.is_scalar() and new.is_matrix():
        return True
    if old.is_matrix() and new.is_scalar():
        return True

    # Matrix vs matrix: check any provable dimension conflicts
    if old.is_matrix() and new.is_matrix():
        if dims_definitely_conflict(old.rows, new.rows):
            return True
        if dims_definitely_conflict(old.cols, new.cols):
            return True

    return False

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
    
    if tag == "matrix":
        line = expr[1]
        rows = expr[2]  # List[List[expr]]
        return eval_matrix_literal(rows, env, warnings, line)

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
        line = expr[1]
        base_expr = expr[2]
        args = expr[3]  # List[Any]

        base_shape = eval_expr(base_expr, env, warnings)

        # If unknown, we can't do much.
        if base_shape.is_unknown():
            return Shape.unknown()

        # Indexing a scalar is suspicious
        if base_shape.is_scalar():
            warnings.append(
                f"Line {line}: Indexing applied to scalar in {pretty_expr(expr)}. "
                f"Treating result as unknown."
            )
            return Shape.unknown()

        # Matrix indexing semantics
        if base_shape.is_matrix():
            m = base_shape.rows
            n = base_shape.cols

            # Linear indexing A(i) turns into a scalar conservatively
            if len(args) == 1:
                return Shape.scalar()

            # 2D indexing A(i,j), A(i,:), A(:,j), A(:,:)
            if len(args) == 2:
                a1, a2 = args
                a1_is_colon = (isinstance(a1, list) and a1[0] == "colon")
                a2_is_colon = (isinstance(a2, list) and a2[0] == "colon")

                if a1_is_colon and a2_is_colon:
                    return Shape.matrix(m, n)
                if a1_is_colon and not a2_is_colon:
                    return Shape.matrix(m, 1)
                if not a1_is_colon and a2_is_colon:
                    return Shape.matrix(1, n)

                # A(i,j)
                return Shape.scalar()

            # Anything else (A(i,j,k)) unsupported
            return Shape.unknown()

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
            return Shape.unknown()
        
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
        if dims_definitely_conflict(left.cols, right.rows):
            msg = (
                f"Line {line}: Dimension mismatch in expression "
                f"{pretty_expr(['*', line, left_expr, right_expr])}: "
                f"inner dims {left.cols} vs {right.rows} (shapes {left} and {right})"
            )

            # If same outer shape, suggest an elementwise multiply
            if (
                left.is_matrix()
                and right.is_matrix()
                and not dims_definitely_conflict(left.rows, right.rows)
                and not dims_definitely_conflict(left.cols, right.cols)
            ):
                msg += ". Did you mean elementwise multiplication (.*)?"

            warnings.append(msg)
            return Shape.unknown()
        
        # Result shape is rows_left x cols_right, even if inner dims are unknown
        return Shape.matrix(left.rows, right.cols)

    # Anything else is unknown
    return Shape.unknown()

def as_matrix_shape(s: Shape) -> Shape:
    """Treat scalar as 1x1 matrix for concatenation."""
    if s.is_scalar():
        return Shape.matrix(1, 1)
    return s


def eval_matrix_literal(
        rows_exprs: List[List[Any]],
        env: Env,
        warnings: List[str],
        line: int
    ) -> Shape:
    had_definite_error = False
    # Empty literal []
    if len(rows_exprs) == 0:
        return Shape.matrix(0, 0)

    row_heights: List[Dim] = []
    row_widths: List[Dim] = []

    for r, row in enumerate(rows_exprs):
        elem_rows: List[Dim] = []
        elem_cols: List[Dim] = []

        for e in row:
            s = as_matrix_shape(eval_expr(e, env, warnings))

            if s.is_unknown():
                elem_rows.append(None)
                elem_cols.append(None)
            elif s.is_matrix():
                elem_rows.append(s.rows)
                elem_cols.append(s.cols)
            else:
                # should not happen, but be safe
                elem_rows.append(None)
                elem_cols.append(None)

        # unify row counts within the row (horizontal concat constraint)
        height = elem_rows[0]
        for rr in elem_rows[1:]:
            if dims_definitely_conflict(height, rr):
                had_definite_error = True
                warnings.append(
                    f"Line {line}: Horizontal concatenation requires equal row counts in row {r+1}; "
                    f"got {height} and {rr} in matrix literal."
                )
            height = join_dim(height, rr)

        width = sum_dims(elem_cols)
        row_heights.append(height)
        row_widths.append(width)

    common_width = row_widths[0]
    for i, w in enumerate(row_widths[1:], start=2):
        if dims_definitely_conflict(common_width, w):
            had_definite_error = True
            warnings.append(
                f"Line {line}: Vertical concatenation requires equal column counts across rows; "
                f"got {common_width} and {w} in matrix literal."
            )
        common_width = join_dim(common_width, w)

    total_height = sum_dims(row_heights)

    if had_definite_error:
        return Shape.unknown()

    return Shape.matrix(total_height, common_width)


def analyze_stmt(
        stmt: Any, 
        env: Env, 
        warnings: List[str]
    ) -> Env:
    tag = stmt[0]

    if tag == "assign":
        assign_line = stmt[1]
        name = stmt[2]
        expr = stmt[3]

        new_shape = eval_expr(expr, env, warnings)
        old_shape = env.get(name)

        if name in env.bindings and shapes_definitely_incompatible(old_shape, new_shape):
            warnings.append(
                f"Line {assign_line}: Variable '{name}' reassigned with incompatible shape "
                f"{new_shape} (previously {old_shape})"
            )

        env.set(name, new_shape)
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