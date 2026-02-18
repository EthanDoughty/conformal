"""Document symbol provider for outline view."""
from __future__ import annotations
from lsprotocol import types
from ir.ir import Program, FunctionDef


def get_document_symbols(ir_prog: Program, source_lines: list[str]) -> list[types.DocumentSymbol]:
    """Extract document symbols from IR program.

    Args:
        ir_prog: Parsed and lowered IR program
        source_lines: Source code lines (for computing ranges)

    Returns:
        List of DocumentSymbol entries (one per function definition)
    """
    symbols: list[types.DocumentSymbol] = []

    for stmt in ir_prog.body:
        if isinstance(stmt, FunctionDef):
            # Compute detail string: (params) -> [outputs]
            params_str = ", ".join(stmt.params)
            outputs_str = ", ".join(stmt.output_vars)
            detail = f"({params_str}) -> [{outputs_str}]"

            # Compute range (1-based IR line -> 0-based LSP line)
            start_line = stmt.line - 1

            # End line: last statement in body, or just the function line if empty
            if stmt.body:
                end_line = max(s.line for s in stmt.body) - 1
            else:
                end_line = start_line

            # Full function range
            full_range = types.Range(
                start=types.Position(line=start_line, character=0),
                end=types.Position(line=end_line, character=len(source_lines[end_line]) if end_line < len(source_lines) else 0)
            )

            # Selection range: just the function definition line
            selection_range = types.Range(
                start=types.Position(line=start_line, character=0),
                end=types.Position(line=start_line, character=len(source_lines[start_line]) if start_line < len(source_lines) else 0)
            )

            symbol = types.DocumentSymbol(
                name=stmt.name,
                kind=types.SymbolKind.Function,
                range=full_range,
                selection_range=selection_range,
                detail=detail
            )

            symbols.append(symbol)

    return symbols
