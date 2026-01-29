# Ethan Doughty
# diagnostics.py

from __future__ import annotations
from typing import List

from runtime.shapes import Shape
from ir.ir import *
# ---------------
# Pretty-printing
# ---------------

def pretty_expr_ir(expr: Expr) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Neg):
        return f"(-{pretty_expr_ir(expr.operand)})"
    if isinstance(expr, Transpose):
        return pretty_expr_ir(expr.operand) + "'"
    if isinstance(expr, Call):
        func = pretty_expr_ir(expr.func)
        args = ", ".join(pretty_expr_ir(a) for a in expr.args)
        return f"{func}({args})"
    if isinstance(expr, Index):
        base = pretty_expr_ir(expr.base)
        args_s = ", ".join(pretty_index_arg_ir(a) for a in expr.args)
        return f"{base}({args_s})"
    if isinstance(expr, MatrixLit):
        return "[matrix]"
    if isinstance(expr, BinOp):
        if expr.op == ":":
            return f"{pretty_expr_ir(expr.left)}:{pretty_expr_ir(expr.right)}"
        return f"({pretty_expr_ir(expr.left)} {expr.op} {pretty_expr_ir(expr.right)})"
    return "<expr>"

def pretty_index_arg_ir(arg: IndexArg) -> str:
    if isinstance(arg, Colon):
        return ":"
    if isinstance(arg, Range):
        return f"{pretty_expr_ir(arg.start)}:{pretty_expr_ir(arg.end)}"
    if isinstance(arg, IndexExpr):
        return pretty_expr_ir(arg.expr)
    return "<idx>"

# ------------------------
# Warning message builders
# ------------------------

def warn_reassign_incompatible(line: int, name: str, new_shape: Shape, old_shape: Shape) -> str:
    return (
        f"Line {line}: Variable '{name}' reassigned with incompatible shape "
        f"{new_shape} (previously {old_shape})"
    )

def warn_suspicious_comparison_matrix_scalar(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> str:
    return (
        f"Line {line}: Suspicious comparison between matrix and scalar in "
        f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right}). "
        f"In MATLAB this is elementwise and may produce a logical matrix."
    )

def warn_matrix_to_matrix_comparison(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> str:
    return (
        f"Line {line}: Matrix-to-matrix comparison in "
        f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right}). "
        f"In MATLAB this is elementwise and may produce a logical matrix."
    )

def warn_logical_op_non_scalar(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> str:
    return (
        f"Line {line}: Logical operator {op} used with non-scalar operand(s) in "
        f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right})."
    )

def warn_indexing_scalar(line: int, expr: Expr) -> str:
    return (
        f"Line {line}: Indexing applied to scalar in {pretty_expr_ir(expr)}. "
        f"Treating result as unknown."
    )

def warn_too_many_indices(line: int, expr: Expr) -> str:
    return (
        f"Line {line}: Too many indices for 2D matrix in {pretty_expr_ir(expr)}. "
        f"Treating result as unknown."
    )

def warn_range_endpoints_must_be_scalar(
    line: int, arg: IndexArg, start_shape: Shape, end_shape: Shape
) -> str:
    return (
        f"Line {line}: Range endpoints in indexing must be scalar; got "
        f"{start_shape} and {end_shape} in {pretty_index_arg_ir(arg)}. Treating result as unknown."
    )

def warn_invalid_range_end_lt_start(line: int, arg: IndexArg) -> str:
    return f"Line {line}: Invalid range in indexing ({pretty_index_arg_ir(arg)}): end < start."

def warn_non_scalar_index_arg(line: int, arg: IndexArg, shape: Shape) -> str:
    return (
        f"Line {line}: Non-scalar index argument {pretty_index_arg_ir(arg)} has shape {shape}. "
        f"Treating indexing result as unknown."
    )

def warn_elementwise_mismatch(line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape) -> str:
    return (
        f"Line {line}: Elementwise {op} mismatch in "
        f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}): {left} vs {right}"
    )

def warn_matmul_mismatch(
    line: int, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape, suggest_elementwise: bool
) -> str:
    msg = (
        f"Line {line}: Dimension mismatch in expression "
        f"({pretty_expr_ir(left_expr)} * {pretty_expr_ir(right_expr)}): "
        f"inner dims {left.cols} vs {right.rows} (shapes {left} and {right})"
    )
    if suggest_elementwise:
        msg += ". Did you mean elementwise multiplication (.*)?"
    return msg