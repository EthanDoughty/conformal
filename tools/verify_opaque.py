#!/usr/bin/env python3
"""Audit opaque (untranslated) statements in conformal-migrate output.

Scans transpiled Python for '# MATLAB:' and '# CONFORMAL:' comment lines
that indicate the translator couldn't fully convert a statement. Reports
per-file counts and a frequency table of the most common untranslated patterns.

Usage:
    python3 tools/verify_opaque.py [OPTIONS]

Options:
    --binary PATH   Path to conformal-migrate binary
    --dir DIR       Directory of .m files (default: dogfood/)
    --file FILE     Audit a single .m file
    --verbose       Show every opaque line, not just summaries
    --jobs N        Parallel workers (default: 8)
    --timeout N     Subprocess timeout in seconds (default: 30)
    --top N         Show top N opaque patterns (default: 30)
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

_binary = None
_timeout = 30
_env = None


def _make_env():
    env = os.environ.copy()
    dotnet_root = os.path.expanduser("~/.dotnet")
    if os.path.isdir(dotnet_root):
        env["DOTNET_ROOT"] = dotnet_root
    env.setdefault("TMPDIR", "/tmp/claude-0/dotnet-tmp")
    return env


# Patterns that indicate untranslated or partially translated code
OPAQUE_RE = re.compile(r"^\s*#\s*(MATLAB|CONFORMAL):\s*(.+)$")


def classify_opaque(tag, content):
    """Classify an opaque comment into a human-readable bucket.

    Returns a short category string for frequency counting.
    """
    content = content.strip()

    if tag == "CONFORMAL":
        # CONFORMAL comments are translator annotations
        if "complex for-loop" in content:
            return "complex for-loop iterator"
        if "complex index-struct" in content:
            return "complex index-struct assignment"
        if "complex field-index" in content:
            return "complex field-index assignment"
        return f"CONFORMAL: {content[:60]}"

    # MATLAB: comments are verbatim untranslated code
    # Try to extract the statement type
    words = content.split()
    if not words:
        return "empty"

    first = words[0]

    # Keyword-based classification
    keyword_stmts = {
        "global", "persistent", "parfor", "spmd", "classdef",
        "enumeration", "events", "arguments",
    }
    if first in keyword_stmts:
        return first

    # Assignment with complex LHS
    if "=" in content and not content.startswith("="):
        lhs = content.split("=")[0].strip()
        # Dynamic field: s.(expr) = ...
        if ".(" in lhs:
            return "dynamic field assign"
        # Multiple dots with indexing
        if "." in lhs and "(" in lhs:
            return "struct-index assign"
        # Cell assignment with braces
        if "{" in lhs:
            return "cell assign"

    # Function call that wasn't mapped
    func_match = re.match(r"^([a-zA-Z_]\w*)\s*\(", content)
    if func_match:
        return f"call: {func_match.group(1)}"

    # Catch-all
    if len(content) > 50:
        return content[:50] + "..."
    return content


def audit_one(m_file_str):
    """Audit a single file. Returns (path, total_lines, opaque_lines, categories)."""
    try:
        result = subprocess.run(
            [_binary, m_file_str, "--stdout"],
            capture_output=True, text=True, timeout=_timeout, env=_env,
        )
    except subprocess.TimeoutExpired:
        return (m_file_str, 0, 0, [], "timeout")
    except FileNotFoundError:
        return (m_file_str, 0, 0, [], "binary not found")

    if result.returncode != 0:
        return (m_file_str, 0, 0, [], "migrate failed")

    code = result.stdout
    if not code.strip():
        return (m_file_str, 0, 0, [], None)

    lines = code.splitlines()
    total = len(lines)
    opaque_entries = []

    for line in lines:
        m = OPAQUE_RE.match(line)
        if m:
            tag = m.group(1)
            content = m.group(2)
            category = classify_opaque(tag, content)
            opaque_entries.append((tag, content, category))

    return (m_file_str, total, len(opaque_entries), opaque_entries, None)


def main():
    global _binary, _timeout, _env

    parser = argparse.ArgumentParser(
        description="Audit opaque statements in conformal-migrate output"
    )
    parser.add_argument("--binary", default="src/migrate/bin/Release/net8.0/conformal-migrate")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--top", type=int, default=30)
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
        sys.exit(1)

    _binary = binary

    # Collect files
    if args.file:
        m_files = [args.file]
    elif args.dir:
        m_files = sorted(str(p) for p in Path(args.dir).rglob("*.m"))
    else:
        project_root = Path(__file__).resolve().parent.parent
        dogfood = project_root / "dogfood"
        if dogfood.exists():
            m_files = sorted(str(p) for p in dogfood.rglob("*.m"))
        else:
            print("Error: no --file or --dir specified and dogfood/ not found", file=sys.stderr)
            sys.exit(1)

    if not m_files:
        print("No .m files found.", file=sys.stderr)
        sys.exit(1)

    total_files = len(m_files)
    total_lines = 0
    total_opaque = 0
    files_with_opaque = 0
    category_counter = Counter()
    failures = 0

    # Per-file results for top-N reporting
    file_opaque_counts = []

    with Pool(processes=args.jobs) as pool:
        for path, n_lines, n_opaque, entries, error in pool.imap_unordered(audit_one, m_files, chunksize=16):
            if error:
                failures += 1
                if args.verbose:
                    print(f"FAIL: {path} -- {error}")
                continue

            total_lines += n_lines
            total_opaque += n_opaque

            if n_opaque > 0:
                files_with_opaque += 1
                file_opaque_counts.append((n_opaque, path))

            for tag, content, category in entries:
                category_counter[category] += 1
                if args.verbose:
                    print(f"  {path}: [{tag}] {content}")

    # Sort files by opaque count descending
    file_opaque_counts.sort(reverse=True)

    # Summary
    clean_files = total_files - files_with_opaque - failures
    pct_clean = clean_files / max(total_files - failures, 1) * 100
    pct_opaque = total_opaque / max(total_lines, 1) * 100

    print()
    print(f"===== Opaque audit: {total_files} files, {total_lines} output lines =====")
    print(f"  Clean files (zero opaque): {clean_files}/{total_files - failures} ({pct_clean:.1f}%)")
    print(f"  Files with opaque stmts:   {files_with_opaque}")
    print(f"  Total opaque lines:        {total_opaque}/{total_lines} ({pct_opaque:.1f}%)")
    if failures:
        print(f"  Migration failures:        {failures}")

    if category_counter:
        print()
        print(f"Top {min(args.top, len(category_counter))} opaque patterns:")
        for category, count in category_counter.most_common(args.top):
            print(f"  {count:>4}x  {category}")

    if file_opaque_counts:
        print()
        print("Files with most opaque statements:")
        for count, path in file_opaque_counts[:10]:
            short = os.path.basename(path)
            print(f"  {count:>4}  {short} ({path})")


if __name__ == "__main__":
    main()
