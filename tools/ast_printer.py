# Ethan Doughty
# ast_printer.py
import sys
from collections import Counter
from typing import Any, List

from frontend.matlab_parser import parse_matlab


def parse_file(path: str):
    """Read a Mini-MATLAB file and return its AST."""
    with open(path, "r") as f:
        src = f.read()
    return parse_matlab(src)


def is_ast_node(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)


def looks_like_args_list(x: Any) -> bool:
    """Indexing/call args are a list of AST nodes (or empty)."""
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True
    return any(is_ast_node(e) for e in x)


def collect_tag_counts(node: Any, counts: Counter) -> None:
    if not isinstance(node, list):
        return
    if len(node) == 0:
        return
    tag = node[0]
    if isinstance(tag, str):
        counts[tag] += 1
    for child in node[1:]:
        collect_tag_counts(child, counts)


def fmt_inline(node: Any) -> str:
    """Small inline formatter for one-line summaries"""
    if not isinstance(node, list):
        return repr(node)
    if len(node) == 0:
        return "[]"

    tag = node[0]
    if tag == "assign":
        return f"{node[2]} = {fmt_inline(node[3])}"
    if tag == "var":
        return f"{node[2]}"
    if tag == "const":
        return f"{node[2]}"
    if tag == "colon":
        return ":"
    if tag == "transpose":
        return f"{fmt_inline(node[2])}'"
    if tag == ":":
        return f"{fmt_inline(node[2])}:{fmt_inline(node[3])}"
    if tag in {"+", "-", "*", "/", ".*", "./", "==", "~=", "<", "<=", ">", ">=", "&&", "||"}:
        return f"({fmt_inline(node[2])} {tag} {fmt_inline(node[3])})"
    if tag == "neg":
        return f"(-{fmt_inline(node[2])})"
    if tag == "index":
        base = fmt_inline(node[2])
        args = node[3]
        if looks_like_args_list(args):
            args_s = ", ".join(fmt_inline(a) for a in args)
        else:
            args_s = fmt_inline(args)
        return f"{base}({args_s})"
    if tag == "call":
        func = fmt_inline(node[2])
        args = node[3]
        args_s = ", ".join(fmt_inline(a) for a in args) if looks_like_args_list(args) else fmt_inline(args)
        return f"{func}({args_s})"
    if tag == "seq":
        return "seq[...]"
    return tag


def fmt_ast(node: Any, indent: int = 0, compact: bool = False) -> str:
    """Pretty-format the internal list-based AST"""
    pad = "  " * indent

    # Atoms
    if not isinstance(node, list):
        return pad + repr(node)

    if len(node) == 0:
        return pad + "[]"

    tag = node[0]

    # Common leaf-ish nodes
    if tag == "var":
        return pad + f"var@{node[1]}({node[2]})"
    if tag == "const":
        return pad + f"const@{node[1]}({node[2]})"
    if tag == "colon":
        return pad + f"colon@{node[1]}(:)"

    # Compact mode
    if compact:
        if tag == "seq":
            out = [pad + "seq["]
            for child in node[1:]:
                out.append(fmt_ast(child, indent + 1, compact=True) + ",")
            out.append(pad + "]")
            return "\n".join(out)

        if len(node) >= 2 and isinstance(node[1], int):
            return pad + f"{tag}@{node[1]} {fmt_inline(node)}"
        return pad + f"{tag} {fmt_inline(node)}"

    # seq (program) prints as a block
    if tag == "seq":
        out = [pad + "seq["]
        for child in node[1:]:
            out.append(fmt_ast(child, indent + 1, compact=compact) + ",")
        out.append(pad + "]")
        return "\n".join(out)

    # assign for readability
    if tag == "assign":
        # ['assign', line, name, expr]
        line = node[1]
        name = node[2]
        expr = node[3]
        out = [
            pad + f"assign@{line}(",
            pad + "  " + f"name={name!r},",
            fmt_ast(expr, indent + 1, compact=compact) + ",",
            pad + ")",
        ]
        return "\n".join(out)

    # call / index because the 4th field is args list
    if tag in {"call", "index"}:
        line = node[1]
        base = node[2]
        args = node[3]

        out = [pad + f"{tag}@{line}("]
        out.append(fmt_ast(base, indent + 1, compact=compact) + ",")

        out.append(pad + "  " + "args=[")
        if looks_like_args_list(args):
            for a in args:
                out.append(fmt_ast(a, indent + 2, compact=compact) + ",")
        else:
            # fallback
            out.append(fmt_ast(args, indent + 2, compact=compact) + ",")
        out.append(pad + "  " + "]")

        out.append(pad + ")")
        return "\n".join(out)

    # Matrix literal node
    if tag == "matrix":
        line = node[1]
        rows: List[List[Any]] = node[2]
        if not rows:
            return pad + f"matrix@{line} []"

        out = [pad + f"matrix@{line} ["]
        for row in rows:
            elems = ", ".join(fmt_inline(e) for e in row)
            out.append(pad + "  " + f"[{elems}]")
        out.append(pad + "]")
        return "\n".join(out)

    # Generic n-ary / statement forms
    if len(node) >= 2 and isinstance(node[1], int):
        line = node[1]
        out = [pad + f"{tag}@{line}("]
        for child in node[2:]:
            out.append(fmt_ast(child, indent + 1, compact=compact) + ",")
        out.append(pad + ")")
        return "\n".join(out)

    # print as tag(children)
    out = [pad + f"{tag}("]
    for child in node[1:]:
        out.append(fmt_ast(child, indent + 1, compact=compact) + ",")
    out.append(pad + ")")
    return "\n".join(out)


def usage() -> None:
    print("Usage: python3 ast_printer.py [--raw] [--compact] <file.m>")
    print("  --raw     Print raw Python list structure (repr)")
    print("  --compact More compact, one-line-ish output")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    raw = False
    compact = False

    while args and args[0].startswith("--"):
        if args[0] == "--raw":
            raw = True
        elif args[0] == "--compact":
            compact = True
        else:
            usage()
        args = args[1:]

    if len(args) != 1:
        usage()

    path = args[0]
    try:
        ast = parse_file(path)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        sys.exit(1)

    print(f"==== AST for {path}")

    if raw:
        # Raw structure is invaluable when debugging the printer itself
        print(repr(ast))
    else:
        print(fmt_ast(ast, compact=compact))

    # Summary counts (quick sanity check)
    counts: Counter = Counter()
    collect_tag_counts(ast, counts)
    print("\n==== Node counts ====")
    for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{tag:>12}: {n}")


if __name__ == "__main__":
    main()