# Ethan Doughty
# ast_printer.py
import sys
from typing import Any, List

from matlab_parser import parse_matlab


def parse_file(path: str):
    """Read a Mini-MATLAB file and return its AST."""
    with open(path, "r") as f:
        src = f.read()
    return parse_matlab(src)


def fmt_ast(node: Any, indent: int = 0) -> str:
    """Pretty-format the internal list-based AST."""
    pad = "  " * indent

    # Atoms
    if not isinstance(node, list):
        return pad + repr(node)

    if len(node) == 0:
        return pad + "[]"

    tag = node[0]

    # Common leaf-ish nodes
    if tag == "var":
        # ['var', line, name]
        return pad + f"var@{node[1]}({node[2]})"
    if tag == "const":
        # ['const', line, value]
        return pad + f"const@{node[1]}({node[2]})"

    # Matrix literal node (new)
    if tag == "matrix":
        # ['matrix', line, rows]
        line = node[1]
        rows: List[List[Any]] = node[2]
        if not rows:
            return pad + f"matrix@{line} []"

        out = [pad + f"matrix@{line} ["]
        for row in rows:
            # Print each row as a bracketed list of exprs
            elems = ", ".join(fmt_ast(e, 0).strip() for e in row)
            out.append(pad + "  " + f"[{elems}]")
        out.append(pad + "]")
        return "\n".join(out)

    # Generic n-ary / statement forms
    # If node[1] is an int, treat it as a line number
    if len(node) >= 2 and isinstance(node[1], int):
        line = node[1]
        out = [pad + f"{tag}@{line}("]
        for child in node[2:]:
            out.append(fmt_ast(child, indent + 1) + ",")
        out.append(pad + ")")
        return "\n".join(out)

    # Sequence / blocks are often ['seq', ...] or bare lists-of-stmts
    out = [pad + f"{tag}("]
    for child in node[1:]:
        out.append(fmt_ast(child, indent + 1) + ",")
    out.append(pad + ")")
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        print("Usage: python ast_printer.py <file.m>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        ast = parse_file(path)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        sys.exit(1)

    print(f"==== AST for {path}")
    print(fmt_ast(ast))


if __name__ == "__main__":
    main()