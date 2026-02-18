"""Hover provider for showing inferred shapes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Set
from lsprotocol import types
from runtime.env import Env


def get_hover(
    env: Env,
    source: str,
    line: int,
    character: int,
    function_registry: Optional[Dict] = None,
    builtins_set: Optional[Set[str]] = None,
    external_functions: Optional[Dict] = None
) -> Optional[types.Hover]:
    """Get hover information for a variable at the given position.

    Args:
        env: Analysis environment containing variable bindings
        source: Full source code text
        line: Zero-indexed line number
        character: Zero-indexed character position in line
        function_registry: Dict of same-file user-defined functions
        builtins_set: Set of known builtin function names
        external_functions: Dict of workspace external functions

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
    word = None
    match_start = 0
    match_end = 0

    for match in re.finditer(identifier_pattern, line_text):
        start, end = match.span()
        if start <= character < end:
            word = match.group(0)
            match_start = start
            match_end = end
            break
    else:
        # No identifier at cursor
        return None

    # Compute hover range
    hover_range = types.Range(
        start=types.Position(line=line, character=match_start),
        end=types.Position(line=line, character=match_end)
    )

    # Look up in environment (variable)
    shape = env.get(word)
    if shape is not None and not shape.is_bottom():
        # Format as Markdown
        shape_str = str(shape)
        hover_text = f"(conformal) `{word}`: `{shape_str}`"

        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=hover_text,
            ),
            range=hover_range
        )

    # Fallback 1: Check function_registry (same-file user-defined functions)
    if function_registry and word in function_registry:
        sig = function_registry[word]
        params_str = ", ".join(sig.params)
        outputs_str = ", ".join(sig.output_vars)
        hover_text = f"(function) `{word}({params_str}) -> [{outputs_str}]`"

        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=hover_text,
            ),
            range=hover_range
        )

    # Fallback 2: Check KNOWN_BUILTINS
    if builtins_set and word in builtins_set:
        hover_text = f"(builtin) `{word}`"

        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=hover_text,
            ),
            range=hover_range
        )

    # Fallback 3: Check external_functions (workspace)
    if external_functions and word in external_functions:
        sig = external_functions[word]
        filename = Path(sig.source_path).name
        hover_text = f"(external) `{word}` from `{filename}`"

        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=hover_text,
            ),
            range=hover_range
        )

    # No match
    return None
