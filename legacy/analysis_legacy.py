# Ethan Doughty
# analysis_legacy.py
from runtime.env import Env
from runtime.shapes import Shape, Dim
from analysis.analysis_core import *
from typing import Any, List, Tuple, Union
from analysis.matrix_literals import infer_matrix_literal_shape, as_matrix_shape

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
        from runtime.env import join_env
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
        rows_exprs = expr[2]  # List[List[expr]]

        shape_rows = [
            [as_matrix_shape(eval_expr(e, env, warnings)) for e in row]
            for row in rows_exprs
        ]
        return infer_matrix_literal_shape(shape_rows, line, warnings)
    
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

                # Determine the extent of each index argument
                r_extent = index_arg_to_extent(a1, env, warnings, line)
                c_extent = index_arg_to_extent(a2, env, warnings, line)

                # If either extent is unknown due to invalid indexing, be strict
                if (r_extent is None and isinstance(a1, list) and a1[0] not in {"colon", "range", ":"}) or \
                (c_extent is None and isinstance(a2, list) and a2[0] not in {"colon", "range", ":"}):
                    return Shape.unknown()

                # Resolve ':' meaning "all rows/cols"
                if isinstance(a1, list) and a1[0] == "colon":
                    r_extent = m
                if isinstance(a2, list) and a2[0] == "colon":
                    c_extent = n

                # Range with unknown extent
                if isinstance(a1, list) and a1[0] in {"range", ":"} and r_extent is None:
                    r_extent = None
                if isinstance(a2, list) and a2[0] in {"range", ":"} and c_extent is None:
                    c_extent = None

                # If both are scalar selections
                if r_extent == 1 and c_extent == 1:
                    return Shape.scalar()

                return Shape.matrix(r_extent, c_extent)
            
            if len(args) > 2:
                warnings.append(
                    f"Line {line}: Too many indices for 2D matrix in {pretty_expr(expr)}. "
                    f"Treating result as unknown."
                )
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
        if elementwise_definitely_mismatch(left, right):
            warnings.append(
                f"Line {line}: Elementwise {op} mismatch: {left} vs {right}"
            )
            return Shape.unknown()
        return elementwise_result_shape(left, right)

    # Matrix multiply
    if op == "*":
        if matmul_definitely_mismatch(left, right):
            msg = (
                f"Line {line}: Dimension mismatch in matrix multiply: "
                f"inner dims {left.cols} vs {right.rows} (shapes {left} and {right})"
            )
            if (left.is_matrix() and right.is_matrix()
                and not dims_definitely_conflict(left.rows, right.rows)
                and not dims_definitely_conflict(left.cols, right.cols)):
                msg += ". Did you mean elementwise multiplication (.*)?"
            warnings.append(msg)
            return Shape.unknown()
        return matmul_result_shape(left, right)

    # Fallback
    return Shape.unknown()

def index_arg_to_extent(
        arg: Any,
        env: Env,
        warnings: List[str],
        line: int
    ) -> Dim:
    """
    Return how many rows/cols this index selects:
      colon -> unknown extent
      scalar expr -> 1
      range a:b -> extent if computable, else None
    """
    tag = arg[0]

    if tag == "colon":
        return None

    # Range inside subscripts: a:b
    if tag in {"range", ":"}:
        start_expr = arg[2]
        end_expr = arg[3]

        start_shape = eval_expr(start_expr, env, warnings)
        end_shape = eval_expr(end_expr, env, warnings)
        if start_shape.is_matrix() or end_shape.is_matrix():
            warnings.append(
                f"Line {line}: Range endpoints in indexing must be scalar; got "
                f"{start_shape} and {end_shape} in {pretty_expr(arg)}. Treating result as unknown."
            )
            return None

        # Interpret endpoints as dimensions
        a = expr_to_dim(start_expr, env)
        b = expr_to_dim(end_expr, env)

        # If both integers, calculate exact extent (b - a) + 1
        if isinstance(a, int) and isinstance(b, int):
            if b < a:
                warnings.append(
                    f"Line {line}: Invalid range in indexing ({pretty_expr(arg)}): end < start."
                )
                return None
            return (b - a) + 1

        # If symbolic or unknown, keep unknown for v0.4
        return None

    # Otherwise, treat as scalar index
    s = eval_expr(arg, env, warnings)
    if s.is_matrix():
        warnings.append(
            f"Line {line}: Non-scalar index argument {pretty_expr(arg)} has shape {s}. "
            f"Treating indexing result as unknown."
        )
        return None

    return 1

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