# Ethan Doughty
# lexer.py
"""MATLAB lexer with context-sensitive single-quote handling.

Tokenizes MATLAB source code into a list of Token objects. Single quotes
are disambiguated between transpose and string delimiters based on the
preceding token kind.
"""

import re
from dataclasses import dataclass
from typing import List

TokenKind = str

@dataclass
class Token:
    kind: TokenKind # "ID", "NUMBER", "FOR", "==", "+"
    value: str # original text
    pos: int # character offset for error messages
    line: int  # line number

# MATLAB keywords in the subset
KEYWORDS = {
    "for", "parfor", "while", "if", "else", "elseif", "end",
    "switch", "case", "otherwise",
    "try", "catch",
    "break", "continue",
    "function", "return",
    "global", "persistent",
}

# Simple tokenization rules (TRANSPOSE handled separately in context-sensitive lexer)
TOKEN_SPEC = [
    ("DQSTRING", r'"(?:[^"]|"")*"'), # double-quoted strings with "" escaping
    ("NUMBER",   r"\d+(?:\.\d*)?"), # ints or floats
    ("ID",       r"[A-Za-z_]\w*"), # identifiers
    ("CONTINUATION", r"\.\.\.[^\n]*\n?"),  # ... line continuation
    ("DOTOP",    r"\.\*|\./|\.\^|\.\'"), # element-wise ops and dot-transpose
    ("OP",       r"==|~=|<=|>=|&&|\|\||[+\-*/<>()=,:\[\];@\\^&|~]"),
    ("DOT",      r"\."), # standalone dot (for recovery from struct/method access)
    ("NEWLINE",  r"\n"),  # only real newlines
    ("SKIP",     r"[ \t]+"), # spaces/tabs
    ("COMMENT",  r"%[^\n]*"), # comments
    ("QUOTE",    r"'"), # single quote (context-sensitive: string or transpose)
    ("CURLYBRACE", r"[{}]"), # curly braces (for recovery from cell arrays)
    ("MISMATCH", r"."), # anything else is an error
]

MASTER_RE = re.compile("|".join(
    f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC
))

def lex(src: str) -> List[Token]:
    """Turn a MATLAB source string into a list of Tokens.

    Context-sensitive lexing for single quotes:
    - After ID/)/]/NUMBER/TRANSPOSE → ' is TRANSPOSE
    - After OP/=/(/,/;/[/NEWLINE/start/keywords → ' starts STRING
    """
    tokens: List[Token] = []
    line = 1
    pos = 0
    prev_kind = None  # Track previous token for context-sensitive quote handling
    bracket_depth = 0  # Track []/{ } nesting for string-vs-transpose in matrix/cell context
    saw_space = False  # Whitespace since last token (for matrix context disambiguation)

    while pos < len(src):
        # Try to match at current position
        m = MASTER_RE.match(src, pos)
        if not m:
            raise SyntaxError(f"Unexpected character {src[pos]!r} at position {pos}")

        kind = m.lastgroup
        value = m.group()
        start_pos = pos

        if kind == "DQSTRING":
            content = value[1:-1].replace('""', '"')
            tokens.append(Token("STRING", content, start_pos, line))
            prev_kind = "STRING"
            saw_space = False
            pos = m.end()
        elif kind == "NUMBER":
            tokens.append(Token("NUMBER", value, start_pos, line))
            prev_kind = "NUMBER"
            saw_space = False
            pos = m.end()
        elif kind == "ID":
            if value in KEYWORDS:
                tokens.append(Token(value.upper(), value, start_pos, line))
                prev_kind = value.upper()
            else:
                tokens.append(Token("ID", value, start_pos, line))
                prev_kind = "ID"
            saw_space = False
            pos = m.end()
        elif kind == "DOTOP":
            tokens.append(Token(kind, value, start_pos, line))
            prev_kind = kind
            saw_space = False
            pos = m.end()
        elif kind == "DOT":
            tokens.append(Token("DOT", value, start_pos, line))
            prev_kind = "DOT"
            saw_space = False
            pos = m.end()
        elif kind == "CURLYBRACE":
            if value == "{":
                bracket_depth += 1
            elif value == "}":
                bracket_depth = max(0, bracket_depth - 1)
            tokens.append(Token(value, value, start_pos, line))
            prev_kind = value
            saw_space = False
            pos = m.end()
        elif kind == "CONTINUATION":
            if "\n" in value:
                line += 1
            pos = m.end()
            # Don't update prev_kind: preserves transpose/string context
        elif kind == "QUOTE":
            # Context-sensitive: is this transpose or string start?
            # Inside [], space before quote means string (MATLAB semantics)
            is_transpose = prev_kind in {"ID", ")", "]", "}", "NUMBER", "TRANSPOSE"}
            if is_transpose and bracket_depth > 0 and saw_space:
                is_transpose = False
            if is_transpose:
                tokens.append(Token("TRANSPOSE", value, start_pos, line))
                prev_kind = "TRANSPOSE"
                saw_space = False
                pos = m.end()
            else:
                # String start: scan ahead for matching closing quote
                end_pos = start_pos + 1
                while end_pos < len(src) and src[end_pos] != "'":
                    if src[end_pos] == "\n":
                        raise SyntaxError(f"Unterminated string at line {line}, pos {start_pos}")
                    end_pos += 1
                if end_pos >= len(src):
                    raise SyntaxError(f"Unterminated string at line {line}, pos {start_pos}")
                # Extract string content (between quotes)
                string_content = src[start_pos+1:end_pos]
                tokens.append(Token("STRING", string_content, start_pos, line))
                prev_kind = "STRING"
                saw_space = False
                pos = end_pos + 1  # Skip past closing quote
        elif kind == "OP":
            if value == "[":
                bracket_depth += 1
            elif value == "]":
                bracket_depth = max(0, bracket_depth - 1)
            tokens.append(Token(value, value, start_pos, line))
            prev_kind = value
            saw_space = False
            pos = m.end()
        elif kind == "NEWLINE":
            tokens.append(Token("NEWLINE", value, start_pos, line))
            if value == "\n":
                line += 1
            prev_kind = "NEWLINE"
            saw_space = False
            pos = m.end()
        elif kind == "SKIP" or kind == "COMMENT":
            saw_space = True
            pos = m.end()
            # Don't update prev_kind for whitespace/comments
        elif kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at {start_pos}")
        else:
            pos = m.end()

    tokens.append(Token("EOF", "", len(src), line))
    return tokens
