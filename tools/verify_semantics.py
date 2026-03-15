#!/usr/bin/env python3
"""Semantic lint of conformal-migrate output using pyflakes.

Goes beyond syntax checking to find undefined names, unused imports,
redefined-but-unused variables, and other semantic issues in transpiled Python.

Usage:
    python3 tools/verify_semantics.py [OPTIONS]

Options:
    --binary PATH   Path to conformal-migrate binary
    --dir DIR       Directory of .m files (default: dogfood/)
    --file FILE     Lint a single .m file
    --verbose       Show clean files too
    --jobs N        Parallel workers (default: 8)
    --timeout N     Subprocess timeout in seconds (default: 30)
    --top N         Show top N issue patterns (default: 30)

Requires: pip install pyflakes
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
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


def classify_issue(message):
    """Classify a pyflakes message into a category for frequency counting."""
    # Undefined name patterns
    m = re.match(r".*undefined name '(\w+)'", message)
    if m:
        name = m.group(1)
        # Group common numpy/scipy/builtin names
        if name in ("np", "numpy"):
            return "undefined: np (missing import)"
        if name.startswith("np_"):
            return f"undefined: {name}"
        return f"undefined: {name}"

    # Unused import
    if "imported but unused" in message:
        m2 = re.search(r"'(\S+)'", message)
        mod = m2.group(1) if m2 else "?"
        return f"unused import: {mod}"

    # Redefined unused
    if "redefined while unused" in message:
        return "redefined unused variable"

    # Local variable referenced before assignment
    if "referenced before assignment" in message:
        return "referenced before assignment"

    # Star imports
    if "import *" in message:
        return "star import"

    # Catch-all
    if len(message) > 60:
        return message[:60] + "..."
    return message


def lint_one(m_file_str):
    """Lint a single file. Returns (path, status, issues, categories).

    status: 'clean', 'issues', 'syntax_error', 'migrate_fail'
    issues: list of (line, message) tuples
    categories: list of category strings (one per issue)
    """
    # Step 1: Transpile
    try:
        result = subprocess.run(
            [_binary, m_file_str, "--stdout"],
            capture_output=True, text=True, timeout=_timeout, env=_env,
        )
    except subprocess.TimeoutExpired:
        return (m_file_str, "migrate_fail", [], [])
    except FileNotFoundError:
        return (m_file_str, "migrate_fail", [], [])

    if result.returncode != 0:
        return (m_file_str, "migrate_fail", [], [])

    code = result.stdout
    if not code.strip():
        return (m_file_str, "clean", [], [])

    # Step 2: Write to temp file (pyflakes needs a file path for good messages)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Step 3: Run pyflakes
        lint_result = subprocess.run(
            [sys.executable, "-m", "pyflakes", tmp_path],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        return (m_file_str, "migrate_fail", [], [])
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not lint_result.stdout.strip():
        return (m_file_str, "clean", [], [])

    # Step 4: Parse pyflakes output
    issues = []
    categories = []
    for line in lint_result.stdout.strip().splitlines():
        # Format: /path/file.py:LINE:COL message
        m = re.match(r".*?:(\d+):(?:\d+:?)?\s*(.+)", line)
        if m:
            lineno = int(m.group(1))
            msg = m.group(2).strip()

            # Skip "unable to detect undefined names" (syntax errors caught elsewhere)
            if "unable to detect" in msg:
                continue

            issues.append((lineno, msg))
            categories.append(classify_issue(msg))

    if not issues:
        return (m_file_str, "clean", [], [])

    return (m_file_str, "issues", issues, categories)


def main():
    global _binary, _timeout, _env

    parser = argparse.ArgumentParser(
        description="Semantic lint of conformal-migrate output using pyflakes"
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
    clean = 0
    with_issues = 0
    migrate_fail = 0
    total_issues = 0
    category_counter = Counter()

    # Files with most issues
    file_issue_counts = []

    with Pool(processes=args.jobs) as pool:
        for path, status, issues, categories in pool.imap_unordered(lint_one, m_files, chunksize=16):
            if status == "clean":
                clean += 1
                if args.verbose:
                    print(f"CLEAN:  {path}")
            elif status == "migrate_fail":
                migrate_fail += 1
                if args.verbose:
                    print(f"FAIL:   {path}")
            else:
                with_issues += 1
                total_issues += len(issues)
                file_issue_counts.append((len(issues), path))

                for cat in categories:
                    category_counter[cat] += 1

                if args.verbose or len(issues) <= 5:
                    # Show individual issues for files with few problems
                    pass

                # Always print files with issues
                undef_count = sum(1 for _, m in issues if "undefined name" in m)
                unused_count = sum(1 for _, m in issues if "imported but unused" in m)
                other_count = len(issues) - undef_count - unused_count
                parts = []
                if undef_count:
                    parts.append(f"{undef_count} undefined")
                if unused_count:
                    parts.append(f"{unused_count} unused import")
                if other_count:
                    parts.append(f"{other_count} other")
                summary = ", ".join(parts)
                print(f"ISSUES: {path} -- {summary}")

                if args.verbose:
                    for lineno, msg in issues:
                        print(f"        line {lineno}: {msg}")

    file_issue_counts.sort(reverse=True)

    analyzed = total_files - migrate_fail
    pct_clean = clean / max(analyzed, 1) * 100

    print()
    print(f"===== Semantic lint: {total_files} files =====")
    print(f"  Clean (no issues):    {clean}/{analyzed} ({pct_clean:.1f}%)")
    print(f"  Files with issues:    {with_issues}")
    print(f"  Total issues:         {total_issues}")
    if migrate_fail:
        print(f"  Migration failures:   {migrate_fail}")

    if category_counter:
        print()
        print(f"Top {min(args.top, len(category_counter))} issue categories:")
        for category, count in category_counter.most_common(args.top):
            print(f"  {count:>4}x  {category}")

    if file_issue_counts:
        print()
        print("Files with most issues:")
        for count, path in file_issue_counts[:10]:
            short = os.path.basename(path)
            print(f"  {count:>4}  {short}")


if __name__ == "__main__":
    main()
