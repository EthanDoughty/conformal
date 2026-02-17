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
}


def to_lsp_diagnostic(d: ConformalDiagnostic, source_lines: list[str]) -> types.Diagnostic:
    """Convert a Conformal Diagnostic to an LSP Diagnostic.

    Args:
        d: Conformal diagnostic with 1-based line numbering
        source_lines: Source code split into lines (for range calculation)

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

    range_ = types.Range(
        start=types.Position(line=line_num, character=0),
        end=types.Position(line=line_num, character=end_char),
    )

    # Determine severity
    if d.code in ERROR_CODES:
        severity = types.DiagnosticSeverity.Error
    elif d.code.startswith("W_UNSUPPORTED_"):
        severity = types.DiagnosticSeverity.Hint
    else:
        severity = types.DiagnosticSeverity.Warning

    # Build diagnostic
    return types.Diagnostic(
        range=range_,
        severity=severity,
        code=d.code if d.code else None,
        source="conformal",
        message=d.message,
    )
