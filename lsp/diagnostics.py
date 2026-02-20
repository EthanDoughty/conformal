"""Convert Conformal Diagnostic objects to LSP Diagnostic objects."""
from __future__ import annotations

from lsprotocol import types
from analysis.diagnostics import Diagnostic as ConformalDiagnostic

# Codes that represent definite errors (dimension mismatches that will crash at runtime)
ERROR_CODES = {
    "W_INNER_DIM_MISMATCH",
    "W_ELEMENTWISE_MISMATCH",
    "W_CONSTRAINT_CONFLICT",
    "W_HORZCAT_ROW_MISMATCH",
    "W_VERTCAT_COL_MISMATCH",
    "W_RESHAPE_MISMATCH",
    "W_INDEX_OUT_OF_BOUNDS",
    "W_DIVISION_BY_ZERO",
    "W_ARITHMETIC_TYPE_MISMATCH",
    "W_TRANSPOSE_TYPE_MISMATCH",
    "W_NEGATE_TYPE_MISMATCH",
    "W_CONCAT_TYPE_MISMATCH",
    "W_INDEX_ASSIGN_TYPE_MISMATCH",
    "W_POSSIBLY_NEGATIVE_DIM",
    "W_FUNCTION_ARG_COUNT_MISMATCH",
    "W_LAMBDA_ARG_COUNT_MISMATCH",
    "W_MULTI_ASSIGN_COUNT_MISMATCH",
    "W_MULTI_ASSIGN_NON_CALL",
    "W_MULTI_ASSIGN_BUILTIN",
    "W_PROCEDURE_IN_EXPR",
    "W_BREAK_OUTSIDE_LOOP",
    "W_CONTINUE_OUTSIDE_LOOP",
    "W_STRICT_MODE",
    "W_MLDIVIDE_DIM_MISMATCH",
    "W_MATRIX_POWER_NON_SQUARE",
}


def to_lsp_diagnostic(d: ConformalDiagnostic, source_lines: list[str], uri: str) -> types.Diagnostic:
    """Convert a Conformal Diagnostic to an LSP Diagnostic.

    Args:
        d: Conformal diagnostic with 1-based line numbering
        source_lines: Source code split into lines (for range calculation)
        uri: Document URI for related_line locations

    Returns:
        LSP Diagnostic with 0-based line numbering
    """
    # Convert to zero-indexed line number
    line_num = d.line - 1

    # Determine range (default: full line)
    # If we have source, span the whole line; otherwise use position 0
    if 0 <= line_num < len(source_lines):
        line_text = source_lines[line_num]
        end_char = len(line_text)
    else:
        end_char = 0

    start_char = d.col - 1 if d.col > 0 else 0  # LSP uses 0-based columns
    range_ = types.Range(
        start=types.Position(line=line_num, character=start_char),
        end=types.Position(line=line_num, character=end_char),
    )

    # Determine severity
    if d.code in ERROR_CODES:
        severity = types.DiagnosticSeverity.Error
    elif d.code.startswith("W_UNSUPPORTED_"):
        severity = types.DiagnosticSeverity.Hint
    else:
        severity = types.DiagnosticSeverity.Warning

    # Add diagnostic tags for unsupported constructs
    tags = None
    if d.code.startswith("W_UNSUPPORTED_"):
        tags = [types.DiagnosticTag.Unnecessary]

    # Add related information if available
    related_information = None
    if d.related_line is not None:
        related_line_num = d.related_line - 1
        if 0 <= related_line_num < len(source_lines):
            related_line_text = source_lines[related_line_num]
            related_end_char = len(related_line_text)
        else:
            related_end_char = 0

        related_range = types.Range(
            start=types.Position(line=related_line_num, character=0),
            end=types.Position(line=related_line_num, character=related_end_char),
        )
        related_location = types.Location(uri=uri, range=related_range)
        related_info = types.DiagnosticRelatedInformation(
            location=related_location,
            message=f"Related: see line {d.related_line}"
        )
        related_information = [related_info]

    # Build diagnostic
    return types.Diagnostic(
        range=range_,
        severity=severity,
        code=d.code if d.code else None,
        source="conformal",
        message=d.message,
        tags=tags,
        related_information=related_information,
    )
