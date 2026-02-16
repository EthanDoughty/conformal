"""Hover provider for showing inferred shapes."""
from __future__ import annotations

import re
from typing import Optional
from lsprotocol import types
from runtime.env import Env


def get_hover(env: Env, source: str, line: int, character: int) -> Optional[types.Hover]:
    """Get hover information for a variable at the given position.

    Args:
        env: Analysis environment containing variable bindings
        source: Full source code text
        line: Zero-indexed line number
        character: Zero-indexed character position in line

    Returns:
        Hover object with shape information, or None if no variable at cursor
    """
    # Split source into lines
    lines = source.split("\n")
    if not (0 <= line < len(lines)):
        return None

    line_text = lines[line]
    if not (0 <= character <= len(line_text)):
        return None

    # Extract word at cursor position
    # Find the identifier that includes the cursor position
    identifier_pattern = r"[A-Za-z_]\w*"
    for match in re.finditer(identifier_pattern, line_text):
        start, end = match.span()
        if start <= character < end:
            word = match.group(0)
            break
    else:
        # No identifier at cursor
        return None

    # Look up in environment
    shape = env.get(word)
    if shape is None or shape.is_bottom():
        return None

    # Format as Markdown
    shape_str = str(shape)
    hover_text = f"(conformal) `{word}`: `{shape_str}`"

    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=hover_text,
        ),
    )
