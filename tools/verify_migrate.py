#!/usr/bin/env python3
"""Verify that conformal-migrate produces syntactically valid Python.

Runs the migrate binary on .m files, pipes output through ast.parse,
and reports failures with categorization.

Usage:
    python3 tools/verify_migrate.py [OPTIONS]

Options:
    --binary PATH   Path to conformal-migrate binary
                    (default: src/migrate/bin/Release/net8.0/conformal-migrate)
    --dir DIR       Directory of .m files to verify (recursive)
                    (default: dogfood/)
    --file FILE     Verify a single .m file
    --verbose       Show passing files too (default: failures only)
    --jobs N        Parallel workers (default: 8)
    --timeout N     Subprocess timeout in seconds (default: 30)
"""

import argparse
import ast
import os
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# Python reserved keywords that MATLAB allows as variable names
PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield",
}

# Resolved at module level so multiprocessing workers can access them
_binary = None
_timeout = 30
_env = None


def find_m_files(directory):
    """Recursively find all .m files, sorted for determinism."""
    return sorted(Path(directory).rglob("*.m"))


def _make_env():
    """Build subprocess environment with DOTNET_ROOT and TMPDIR."""
    env = os.environ.copy()
    dotnet_root = os.path.expanduser("~/.dotnet")
    if os.path.isdir(dotnet_root):
        env["DOTNET_ROOT"] = dotnet_root
    env.setdefault("TMPDIR", "/tmp/claude-0/dotnet-tmp")
    return env


def check_syntax(python_code):
    """Run ast.parse on Python code. Return (ok, error_info)."""
    try:
        ast.parse(python_code)
        return True, None
    except SyntaxError as e:
        return False, {
            "line": e.lineno,
            "col": e.offset,
            "msg": e.msg,
            "text": (e.text or "").rstrip(),
        }


def categorize_error(error_info):
    """Categorize a SyntaxError into a human-readable bucket."""
    text = error_info["text"]
    msg = error_info["msg"]

    for kw in PYTHON_KEYWORDS:
        if text.startswith(f"{kw} =") or text.startswith(f"{kw}="):
            return f"reserved keyword as variable: {kw}"
        if f"({kw}," in text or f"({kw})" in text or f", {kw}," in text or f", {kw})" in text:
            return f"reserved keyword as parameter: {kw}"
        if f".{kw}" in text:
            idx = text.index(f".{kw}")
            end = idx + len(kw) + 1
            if end >= len(text) or not text[end].isalnum():
                return f"reserved keyword as field: {kw}"

    if "indent" in msg.lower():
        return f"indentation: {msg}"
    if "unexpected EOF" in msg or "was never closed" in msg:
        return f"unbalanced: {msg}"
    return msg


def verify_one(m_file_str):
    """Verify a single file. Returns (path, status, category, error_info).
    status is 'ok', 'syntax_fail', or 'migrate_fail'."""
    try:
        result = subprocess.run(
            [_binary, m_file_str, "--stdout"],
            capture_output=True, text=True, timeout=_timeout, env=_env,
        )
    except subprocess.TimeoutExpired:
        return (m_file_str, "migrate_fail", "timeout", None)
    except FileNotFoundError:
        return (m_file_str, "migrate_fail", "binary not found", None)

    if result.returncode != 0:
        return (m_file_str, "migrate_fail", result.stderr.strip(), None)

    code = result.stdout
    if not code.strip():
        return (m_file_str, "ok", None, None)

    ok, error_info = check_syntax(code)
    if ok:
        return (m_file_str, "ok", None, None)
    else:
        cat = categorize_error(error_info)
        return (m_file_str, "syntax_fail", cat, error_info)


def main():
    global _binary, _timeout, _env

    parser = argparse.ArgumentParser(
        description="Verify conformal-migrate produces valid Python"
    )
    parser.add_argument("--binary", default="src/migrate/bin/Release/net8.0/conformal-migrate")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    _timeout = args.timeout
    _env = _make_env()

    binary = args.binary
    if not os.path.isabs(binary):
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / binary
        if candidate.exists():
            binary = str(candidate)

    if not os.path.exists(binary):
        print(f"Error: binary not found at {binary}", file=sys.stderr)
        print("Build with: dotnet build src/migrate/ConformalMigrate.fsproj -c Release", file=sys.stderr)
        sys.exit(1)

    _binary = binary

    # Collect files
    if args.file:
        m_files = [args.file]
    elif args.dir:
        m_files = [str(p) for p in find_m_files(args.dir)]
    else:
        project_root = Path(__file__).resolve().parent.parent
        dogfood = project_root / "dogfood"
        if dogfood.exists():
            m_files = [str(p) for p in find_m_files(dogfood)]
        else:
            print("Error: no --file or --dir specified and dogfood/ not found", file=sys.stderr)
            sys.exit(1)

    if not m_files:
        print("No .m files found.", file=sys.stderr)
        sys.exit(1)

    total = len(m_files)
    syntax_ok = 0
    syntax_fail = 0
    migrate_fail = 0
    error_categories = Counter()

    # Run in parallel
    with Pool(processes=args.jobs) as pool:
        for path, status, category, error_info in pool.imap_unordered(verify_one, m_files, chunksize=16):
            if status == "ok":
                syntax_ok += 1
                if args.verbose:
                    print(f"SYNTAX OK:    {path}")
            elif status == "migrate_fail":
                migrate_fail += 1
                if args.verbose:
                    print(f"MIGRATE FAIL: {path} -- {category}")
            else:
                syntax_fail += 1
                error_categories[category] += 1
                line = error_info["line"] if error_info else "?"
                msg = error_info["msg"] if error_info else category
                print(f"SYNTAX FAIL:  {path} -- line {line}: {msg}")
                if error_info and error_info["text"]:
                    print(f"              {error_info['text']}")

    # Summary
    print()
    print(f"===== Verification: {syntax_ok}/{total} syntax OK, "
          f"{syntax_fail} syntax errors, {migrate_fail} migrate failures =====")

    if error_categories:
        print()
        print("Error categories:")
        for category, count in error_categories.most_common():
            print(f"  {count:>4}x  {category}")

    sys.exit(0 if syntax_fail == 0 else 1)


if __name__ == "__main__":
    main()
