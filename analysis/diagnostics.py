# Ethan Doughty
# diagnostics.py

from __future__ import annotations

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
    if isinstance(expr, StringLit):
        return f"'{expr.value}'"
    if isinstance(expr, Neg):
        return f"(-{pretty_expr_ir(expr.operand)})"
    if isinstance(expr, Transpose):
        return pretty_expr_ir(expr.operand) + "'"
    if isinstance(expr, Lambda):
        params = ", ".join(expr.params)
        return f"@({params}) <body>"
    if isinstance(expr, FuncHandle):
        return f"@{expr.name}"
    if isinstance(expr, Apply):
        base = pretty_expr_ir(expr.base) if isinstance(expr.base, Var) else "<expr>"
        args_s = ", ".join(pretty_index_arg_ir(a) for a in expr.args)
        return f"{base}({args_s})"
    if isinstance(expr, CurlyApply):
        base = pretty_expr_ir(expr.base) if isinstance(expr.base, Var) else "<expr>"
        args_s = ", ".join(pretty_index_arg_ir(a) for a in expr.args)
        return f"{base}{{{args_s}}}"
    if isinstance(expr, MatrixLit):
        return "[matrix]"
    if isinstance(expr, CellLit):
        return "{cell}"
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

def has_unsupported(diags: list[str]) -> bool:
    """Check if any diagnostic is an unsupported statement warning.

    Args:
        diags: List of diagnostic messages

    Returns:
        True if any diagnostic starts with W_UNSUPPORTED_
    """
    return any(d.startswith("W_UNSUPPORTED_") for d in diags)

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


def warn_unsupported_stmt(line: int, raw: str, targets: list[str]) -> str:
    """Warning for unsupported/opaque statement.

    Args:
        line: Source line number
        raw: Original statement text (if available)
        targets: Variables that will be set to unknown

    Returns:
        Warning message string with stable code token W_UNSUPPORTED_STMT
    """
    target_str = ", ".join(targets) if targets else "(none)"
    raw_str = f" '{raw}'" if raw else ""
    return f"W_UNSUPPORTED_STMT line={line} targets={target_str}{raw_str}"

def warn_unknown_function(line: int, name: str) -> str:
    """Warning for unrecognized function call.

    Args:
        line: Source line number
        name: Name of the unrecognized function

    Returns:
        Warning message string with code W_UNKNOWN_FUNCTION
    """
    return f"Line {line}: Function '{name}' is not recognized; treating result as unknown [W_UNKNOWN_FUNCTION]"

def warn_unsupported_multi_assign(line: int) -> str:
    """Warning for unsupported destructuring assignment (Phase A stub).

    Args:
        line: Source line number

    Returns:
        Warning message string with code W_UNSUPPORTED_MULTI_ASSIGN
    """
    return f"W_UNSUPPORTED_MULTI_ASSIGN line {line}: Destructuring assignment not yet supported (Phase C)"

def warn_function_arg_count_mismatch(line: int, func_name: str, expected: int, got: int) -> str:
    """Warning for function called with wrong number of arguments."""
    return f"W_FUNCTION_ARG_COUNT_MISMATCH line {line}: function {func_name} expects {expected} arguments, got {got}"

def warn_recursive_function(line: int, func_name: str) -> str:
    """Warning for recursive function call (not supported, returns unknown)."""
    return f"W_RECURSIVE_FUNCTION line {line}: recursive call to {func_name} not supported (returns unknown)"

def warn_procedure_in_expr(line: int, func_name: str) -> str:
    """Warning for procedure (no return value) used in expression context."""
    return f"W_PROCEDURE_IN_EXPR line {line}: procedure {func_name} has no return value, cannot be used in expression"

def warn_multi_assign_non_call(line: int) -> str:
    """Warning for destructuring assignment with non-function-call RHS."""
    return f"W_MULTI_ASSIGN_NON_CALL line {line}: destructuring assignment requires function call on RHS"

def warn_multi_assign_builtin(line: int, func_name: str) -> str:
    """Warning for destructuring assignment with builtin function (no multi-return)."""
    return f"W_MULTI_ASSIGN_BUILTIN line {line}: builtin {func_name} does not support multiple returns"

def warn_multi_assign_count_mismatch(line: int, func_name: str, expected: int, got: int) -> str:
    """Warning for destructuring assignment target count mismatch."""
    return f"W_MULTI_ASSIGN_COUNT_MISMATCH line {line}: function {func_name} returns {expected} values, got {got} targets"

def warn_string_arithmetic(line: int, op: str, left_shape: Shape, right_shape: Shape) -> str:
    """Warning for invalid string arithmetic (string + matrix/scalar)."""
    return f"W_STRING_ARITHMETIC line {line}: Invalid string arithmetic ({left_shape} {op} {right_shape})"

def warn_struct_field_not_found(line: int, field: str, struct_shape: Shape) -> str:
    """Warning for accessing non-existent struct field."""
    return f"W_STRUCT_FIELD_NOT_FOUND line {line}: Field '{field}' not found in {struct_shape}"

def warn_field_access_non_struct(line: int, base_shape: Shape) -> str:
    """Warning for field access on non-struct value."""
    return f"W_FIELD_ACCESS_NON_STRUCT line {line}: Field access on non-struct value ({base_shape})"

def warn_curly_indexing_non_cell(line: int, base_shape: Shape) -> str:
    """Warning for curly indexing on non-cell value."""
    return f"W_CURLY_INDEXING_NON_CELL line {line}: Curly indexing on non-cell value ({base_shape})"

def warn_cell_assign_non_cell(line: int, base_name: str, base_shape: Shape) -> str:
    """Warning for cell assignment on non-cell variable."""
    return f"W_CELL_ASSIGN_NON_CELL line {line}: Cell assignment to non-cell variable '{base_name}' ({base_shape})"

def warn_return_outside_function(line: int) -> str:
    """Warning for return statement outside function body."""
    return f"W_RETURN_OUTSIDE_FUNCTION line {line}: return statement outside function body"

def warn_break_outside_loop(line: int) -> str:
    """Warning for break statement outside loop (v0.11.1 will validate)."""
    return f"W_BREAK_OUTSIDE_LOOP line {line}: break statement outside loop (treated as no-op)"

def warn_continue_outside_loop(line: int) -> str:
    """Warning for continue statement outside loop (v0.11.1 will validate)."""
    return f"W_CONTINUE_OUTSIDE_LOOP line {line}: continue statement outside loop (treated as no-op)"

def warn_lambda_call_approximate(line: int, var_name: str) -> str:
    """Warning for calling function handle variable (v0.12.0 limitation)."""
    return f"W_LAMBDA_CALL_APPROXIMATE line {line}: Calling function handle '{var_name}' returns unknown (body analysis deferred to v0.12.1)"

def warn_lambda_arg_count_mismatch(line: int, expected: int, got: int) -> str:
    """Warning for lambda called with wrong number of arguments."""
    return f"W_LAMBDA_ARG_COUNT_MISMATCH line {line}: lambda expects {expected} arguments, got {got}"

def warn_recursive_lambda(line: int) -> str:
    """Warning for recursive lambda call (not supported, returns unknown)."""
    return f"W_RECURSIVE_LAMBDA line {line}: recursive lambda call not supported (returns unknown)"

def warn_end_outside_indexing(line: int) -> str:
    """Warning for 'end' keyword used outside indexing context."""
    return f"W_END_OUTSIDE_INDEXING line {line}: 'end' keyword only valid inside indexing expressions"
