# Ethan Doughty
# diagnostics.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from runtime.shapes import Shape
from ir.ir import *

# ---------------
# Diagnostic dataclass
# ---------------

@dataclass(frozen=True)
class Diagnostic:
    """Structured warning/error diagnostic.

    Fields:
        line: Source line number
        code: Warning code (e.g. "W_INNER_DIM_MISMATCH"), empty string for unnamed
        message: Human-readable message (no line number prefix)
        related_line: Optional related line (e.g., call site for body warnings)
    """
    line: int
    code: str
    message: str
    related_line: Optional[int] = None

    def __str__(self) -> str:
        """Backward-compatible string rendering for test expectations."""
        # Special case: W_UNSUPPORTED_STMT uses line= instead of line
        if self.code == "W_UNSUPPORTED_STMT":
            return f"{self.code} line={self.line} {self.message}"

        if self.code:
            return f"{self.code} line {self.line}: {self.message}"
        else:
            return f"Line {self.line}: {self.message}"

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

def has_unsupported(diags: list['Diagnostic']) -> bool:
    """Check if any diagnostic is an unsupported statement warning.

    Args:
        diags: List of diagnostics

    Returns:
        True if any diagnostic has code starting with W_UNSUPPORTED_
    """
    return any(d.code.startswith("W_UNSUPPORTED_") for d in diags)

def warn_reassign_incompatible(line: int, name: str, new_shape: Shape, old_shape: Shape) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_REASSIGN_INCOMPATIBLE",
        message=f"Variable '{name}' reassigned with incompatible shape {new_shape} (previously {old_shape})"
    )

def warn_suspicious_comparison_matrix_scalar(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_SUSPICIOUS_COMPARISON",
        message=(
            f"Suspicious comparison between matrix and scalar in "
            f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right}). "
            f"In MATLAB this is elementwise and may produce a logical matrix."
        )
    )

def warn_matrix_to_matrix_comparison(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_MATRIX_COMPARISON",
        message=(
            f"Matrix-to-matrix comparison in "
            f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right}). "
            f"In MATLAB this is elementwise and may produce a logical matrix."
        )
    )

def warn_logical_op_non_scalar(
    line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape
) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_LOGICAL_OP_NON_SCALAR",
        message=(
            f"Logical operator {op} used with non-scalar operand(s) in "
            f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}) ({left} vs {right})."
        )
    )

def warn_indexing_scalar(line: int, expr: Expr) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_INDEXING_SCALAR",
        message=f"Indexing applied to scalar in {pretty_expr_ir(expr)}. Treating result as unknown."
    )

def warn_too_many_indices(line: int, expr: Expr) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_TOO_MANY_INDICES",
        message=f"Too many indices for 2D matrix in {pretty_expr_ir(expr)}. Treating result as unknown."
    )

def warn_range_endpoints_must_be_scalar(
    line: int, arg: IndexArg, start_shape: Shape, end_shape: Shape
) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_RANGE_NON_SCALAR",
        message=(
            f"Range endpoints in indexing must be scalar; got "
            f"{start_shape} and {end_shape} in {pretty_index_arg_ir(arg)}. Treating result as unknown."
        )
    )

def warn_invalid_range_end_lt_start(line: int, arg: IndexArg) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_INVALID_RANGE",
        message=f"Invalid range in indexing ({pretty_index_arg_ir(arg)}): end < start."
    )

def warn_non_scalar_index_arg(line: int, arg: IndexArg, shape: Shape) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_NON_SCALAR_INDEX",
        message=(
            f"Non-scalar index argument {pretty_index_arg_ir(arg)} has shape {shape}. "
            f"Treating indexing result as unknown."
        )
    )

def warn_elementwise_mismatch(line: int, op: str, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape) -> Diagnostic:
    return Diagnostic(
        line=line,
        code="W_ELEMENTWISE_MISMATCH",
        message=(
            f"Elementwise {op} mismatch in "
            f"({pretty_expr_ir(left_expr)} {op} {pretty_expr_ir(right_expr)}): {left} vs {right}"
        )
    )

def warn_matmul_mismatch(
    line: int, left_expr: Expr, right_expr: Expr, left: Shape, right: Shape, suggest_elementwise: bool
) -> Diagnostic:
    msg = (
        f"Dimension mismatch in expression "
        f"({pretty_expr_ir(left_expr)} * {pretty_expr_ir(right_expr)}): "
        f"inner dims {left.cols} vs {right.rows} (shapes {left} and {right})"
    )
    if suggest_elementwise:
        msg += ". Did you mean elementwise multiplication (.*)?"
    return Diagnostic(
        line=line,
        code="W_INNER_DIM_MISMATCH",
        message=msg
    )


def warn_unsupported_stmt(line: int, raw: str, targets: list[str]) -> Diagnostic:
    """Warning for unsupported/opaque statement.

    Args:
        line: Source line number
        raw: Original statement text (if available)
        targets: Variables that will be set to unknown

    Returns:
        Diagnostic with code W_UNSUPPORTED_STMT
    """
    target_str = ", ".join(targets) if targets else "(none)"
    raw_str = f" '{raw}'" if raw else ""
    # This warning uses a unique format: "W_UNSUPPORTED_STMT line=N targets=..."
    # We need to override the __str__ format for backward compatibility
    # Store the special format in the message itself
    return Diagnostic(
        line=line,
        code="W_UNSUPPORTED_STMT",
        message=f"targets={target_str}{raw_str}"
    )

def warn_unknown_function(line: int, name: str) -> Diagnostic:
    """Warning for unrecognized function call.

    Args:
        line: Source line number
        name: Name of the unrecognized function

    Returns:
        Diagnostic with code W_UNKNOWN_FUNCTION
    """
    return Diagnostic(
        line=line,
        code="W_UNKNOWN_FUNCTION",
        message=f"Function '{name}' is not recognized; treating result as unknown"
    )

def warn_unsupported_multi_assign(line: int) -> Diagnostic:
    """Warning for unsupported destructuring assignment (Phase A stub).

    Args:
        line: Source line number

    Returns:
        Diagnostic with code W_UNSUPPORTED_MULTI_ASSIGN
    """
    return Diagnostic(
        line=line,
        code="W_UNSUPPORTED_MULTI_ASSIGN",
        message="Destructuring assignment not yet supported (Phase C)"
    )

def warn_function_arg_count_mismatch(line: int, func_name: str, expected: int, got: int) -> Diagnostic:
    """Warning for function called with wrong number of arguments."""
    return Diagnostic(
        line=line,
        code="W_FUNCTION_ARG_COUNT_MISMATCH",
        message=f"function {func_name} expects {expected} arguments, got {got}"
    )

def warn_recursive_function(line: int, func_name: str) -> Diagnostic:
    """Warning for recursive function call (not supported, returns unknown)."""
    return Diagnostic(
        line=line,
        code="W_RECURSIVE_FUNCTION",
        message=f"recursive call to {func_name} not supported (returns unknown)"
    )

def warn_procedure_in_expr(line: int, func_name: str) -> Diagnostic:
    """Warning for procedure (no return value) used in expression context."""
    return Diagnostic(
        line=line,
        code="W_PROCEDURE_IN_EXPR",
        message=f"procedure {func_name} has no return value, cannot be used in expression"
    )

def warn_multi_assign_non_call(line: int) -> Diagnostic:
    """Warning for destructuring assignment with non-function-call RHS."""
    return Diagnostic(
        line=line,
        code="W_MULTI_ASSIGN_NON_CALL",
        message="destructuring assignment requires function call on RHS"
    )

def warn_multi_assign_builtin(line: int, func_name: str) -> Diagnostic:
    """Warning for destructuring assignment with builtin function (no multi-return)."""
    return Diagnostic(
        line=line,
        code="W_MULTI_ASSIGN_BUILTIN",
        message=f"builtin {func_name} does not support multiple returns"
    )

def warn_multi_assign_count_mismatch(line: int, func_name: str, expected: int, got: int) -> Diagnostic:
    """Warning for destructuring assignment target count mismatch."""
    return Diagnostic(
        line=line,
        code="W_MULTI_ASSIGN_COUNT_MISMATCH",
        message=f"function {func_name} returns {expected} values, got {got} targets"
    )

def warn_string_arithmetic(line: int, op: str, left_shape: Shape, right_shape: Shape) -> Diagnostic:
    """Warning for invalid string arithmetic (string + matrix/scalar)."""
    return Diagnostic(
        line=line,
        code="W_STRING_ARITHMETIC",
        message=f"Invalid string arithmetic ({left_shape} {op} {right_shape})"
    )

def warn_struct_field_not_found(line: int, field: str, struct_shape: Shape) -> Diagnostic:
    """Warning for accessing non-existent struct field."""
    return Diagnostic(
        line=line,
        code="W_STRUCT_FIELD_NOT_FOUND",
        message=f"Field '{field}' not found in {struct_shape}"
    )

def warn_field_access_non_struct(line: int, base_shape: Shape) -> Diagnostic:
    """Warning for field access on non-struct value."""
    return Diagnostic(
        line=line,
        code="W_FIELD_ACCESS_NON_STRUCT",
        message=f"Field access on non-struct value ({base_shape})"
    )

def warn_curly_indexing_non_cell(line: int, base_shape: Shape) -> Diagnostic:
    """Warning for curly indexing on non-cell value."""
    return Diagnostic(
        line=line,
        code="W_CURLY_INDEXING_NON_CELL",
        message=f"Curly indexing on non-cell value ({base_shape})"
    )

def warn_cell_assign_non_cell(line: int, base_name: str, base_shape: Shape) -> Diagnostic:
    """Warning for cell assignment on non-cell variable."""
    return Diagnostic(
        line=line,
        code="W_CELL_ASSIGN_NON_CELL",
        message=f"Cell assignment to non-cell variable '{base_name}' ({base_shape})"
    )

def warn_return_outside_function(line: int) -> Diagnostic:
    """Warning for return statement outside function body."""
    return Diagnostic(
        line=line,
        code="W_RETURN_OUTSIDE_FUNCTION",
        message="return statement outside function body"
    )

def warn_break_outside_loop(line: int) -> Diagnostic:
    """Warning for break statement outside loop (v0.11.1 will validate)."""
    return Diagnostic(
        line=line,
        code="W_BREAK_OUTSIDE_LOOP",
        message="break statement outside loop (treated as no-op)"
    )

def warn_continue_outside_loop(line: int) -> Diagnostic:
    """Warning for continue statement outside loop (v0.11.1 will validate)."""
    return Diagnostic(
        line=line,
        code="W_CONTINUE_OUTSIDE_LOOP",
        message="continue statement outside loop (treated as no-op)"
    )

def warn_lambda_call_approximate(line: int, var_name: str) -> Diagnostic:
    """Warning for calling function handle variable (v0.12.0 limitation)."""
    return Diagnostic(
        line=line,
        code="W_LAMBDA_CALL_APPROXIMATE",
        message=f"Calling function handle '{var_name}' returns unknown (body analysis deferred to v0.12.1)"
    )

def warn_lambda_arg_count_mismatch(line: int, expected: int, got: int) -> Diagnostic:
    """Warning for lambda called with wrong number of arguments."""
    return Diagnostic(
        line=line,
        code="W_LAMBDA_ARG_COUNT_MISMATCH",
        message=f"lambda expects {expected} arguments, got {got}"
    )

def warn_recursive_lambda(line: int) -> Diagnostic:
    """Warning for recursive lambda call (not supported, returns unknown)."""
    return Diagnostic(
        line=line,
        code="W_RECURSIVE_LAMBDA",
        message="recursive lambda call not supported (returns unknown)"
    )

def warn_end_outside_indexing(line: int) -> Diagnostic:
    """Warning for 'end' keyword used outside indexing context."""
    return Diagnostic(
        line=line,
        code="W_END_OUTSIDE_INDEXING",
        message="'end' keyword only valid inside indexing expressions"
    )

def warn_constraint_conflict(line: int, var_name: str, value: int, other_dim, source_line: int) -> Diagnostic:
    """Warning for constraint conflict detected during validation.

    Args:
        line: Line where binding occurs
        var_name: Variable name being bound
        value: Concrete value being assigned
        other_dim: The conflicting dimension from constraint
        source_line: Line where constraint was created

    Returns:
        Diagnostic with code W_CONSTRAINT_CONFLICT
    """
    return Diagnostic(
        line=line,
        code="W_CONSTRAINT_CONFLICT",
        message=f"{var_name}={value} conflicts with {var_name}=={other_dim} (from line {source_line})"
    )
