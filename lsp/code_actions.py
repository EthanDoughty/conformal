"""Code actions (quick fixes) for Conformal diagnostics."""
from __future__ import annotations

import re
from typing import List

from lsprotocol import types


def code_actions_for_diagnostic(
    diagnostic: types.Diagnostic,
    uri: str,
    source_lines: list[str],
) -> List[types.CodeAction]:
    """Generate code actions for a single diagnostic.

    Args:
        diagnostic: LSP diagnostic to generate fixes for
        uri: Document URI
        source_lines: Source code split into lines

    Returns:
        List of CodeAction quick fixes (may be empty)
    """
    actions: List[types.CodeAction] = []
    code = diagnostic.code
    line_num = diagnostic.range.start.line

    if line_num < 0 or line_num >= len(source_lines):
        return actions

    line_text = source_lines[line_num]

    if code == "W_INNER_DIM_MISMATCH" and "elementwise multiplication" in diagnostic.message:
        # Replace * with .* (but not existing .*)
        new_text = re.sub(r'(?<!\.)\*', '.*', line_text)
        if new_text != line_text:
            edit = types.WorkspaceEdit(
                changes={
                    uri: [
                        types.TextEdit(
                            range=types.Range(
                                start=types.Position(line=line_num, character=0),
                                end=types.Position(line=line_num, character=len(line_text)),
                            ),
                            new_text=new_text,
                        )
                    ]
                }
            )
            actions.append(
                types.CodeAction(
                    title="Replace * with .* (elementwise)",
                    kind=types.CodeActionKind.QuickFix,
                    diagnostics=[diagnostic],
                    edit=edit,
                )
            )

    elif code == "W_LOGICAL_OP_NON_SCALAR":
        # Replace && with & or || with |
        if "&&" in line_text:
            new_text = line_text.replace("&&", "&")
            edit = types.WorkspaceEdit(
                changes={
                    uri: [
                        types.TextEdit(
                            range=types.Range(
                                start=types.Position(line=line_num, character=0),
                                end=types.Position(line=line_num, character=len(line_text)),
                            ),
                            new_text=new_text,
                        )
                    ]
                }
            )
            actions.append(
                types.CodeAction(
                    title="Replace && with & (elementwise)",
                    kind=types.CodeActionKind.QuickFix,
                    diagnostics=[diagnostic],
                    edit=edit,
                )
            )
        if "||" in line_text:
            new_text = line_text.replace("||", "|")
            edit = types.WorkspaceEdit(
                changes={
                    uri: [
                        types.TextEdit(
                            range=types.Range(
                                start=types.Position(line=line_num, character=0),
                                end=types.Position(line=line_num, character=len(line_text)),
                            ),
                            new_text=new_text,
                        )
                    ]
                }
            )
            actions.append(
                types.CodeAction(
                    title="Replace || with | (elementwise)",
                    kind=types.CodeActionKind.QuickFix,
                    diagnostics=[diagnostic],
                    edit=edit,
                )
            )

    return actions
