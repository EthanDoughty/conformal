#!/usr/bin/env python3
"""Compare F# and Python parsers on all .m test files.

Runs both parsers on every test file and compares their IR JSON output.
Reports per-file status and a summary line.

Usage:
    python3 tools/validate_fsharp.py
    python3 tools/validate_fsharp.py --verbose   # show diffs for mismatches
    python3 tools/validate_fsharp.py --limit N   # process at most N files
"""

import argparse
import glob
import json
import os
import sys

# Ensure repo root is on sys.path regardless of where this script is invoked
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontend.matlab_parser import parse_matlab
from frontend.ir_json import ir_to_json, ir_from_json
from frontend.parse_fsharp import parse_matlab_fsharp


def _normalize_json(obj) -> str:
    """Produce a canonical JSON string for comparison.

    Sorts object keys and uses compact separators so that whitespace
    differences do not cause false mismatches.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def _parse_python_json(filepath: str) -> tuple[str, str | None]:
    """Return (normalized_json, error_message) for the Python parser."""
    try:
        src = open(filepath, encoding='utf-8', errors='replace').read()
        prog = parse_matlab(src)
        raw = ir_to_json(prog)
        obj = json.loads(raw)
        return _normalize_json(obj), None
    except Exception as e:
        return '', f"{type(e).__name__}: {e}"


def _parse_fsharp_json(filepath: str) -> tuple[str, str | None]:
    """Return (normalized_json, error_message) for the F# parser."""
    try:
        src = open(filepath, encoding='utf-8', errors='replace').read()
        # Pass filepath so the bridge avoids an unnecessary temp-file round-trip
        prog = parse_matlab_fsharp(src, filepath=filepath)
        raw = ir_to_json(prog)
        obj = json.loads(raw)
        return _normalize_json(obj), None
    except Exception as e:
        return '', f"{type(e).__name__}: {e}"


def _short_diff(py_json: str, fs_json: str) -> str:
    """Return a short human-readable summary of the first divergence."""
    import difflib
    py_lines = py_json.splitlines()
    fs_lines = fs_json.splitlines()
    diff = list(difflib.unified_diff(py_lines, fs_lines,
                                     fromfile='python', tofile='fsharp',
                                     lineterm='', n=2))
    if not diff:
        return "(no textual difference)"
    # Return up to 20 lines so the report is readable
    return '\n'.join(diff[:20])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare F# and Python parsers on all .m test files."
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show unified diff for mismatches')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N files (0 = all)')
    args = parser.parse_args()

    test_files = sorted(glob.glob(
        os.path.join(_REPO_ROOT, 'tests', '**', '*.m'), recursive=True
    ))

    if args.limit > 0:
        test_files = test_files[:args.limit]

    total = len(test_files)
    identical = 0
    diffs = 0
    crashes = 0  # either parser crashed

    for filepath in test_files:
        rel = os.path.relpath(filepath, _REPO_ROOT)
        py_json, py_err = _parse_python_json(filepath)
        fs_json, fs_err = _parse_fsharp_json(filepath)

        if py_err or fs_err:
            crashes += 1
            parts = []
            if py_err:
                parts.append(f"  python: {py_err}")
            if fs_err:
                parts.append(f"  fsharp: {fs_err}")
            print(f"CRASH  {rel}")
            for p in parts:
                print(p)
        elif py_json == fs_json:
            identical += 1
            # Only print in verbose mode to keep output tidy
            if args.verbose:
                print(f"OK     {rel}")
        else:
            diffs += 1
            print(f"DIFF   {rel}")
            if args.verbose:
                print(_short_diff(py_json, fs_json))

    print()
    print(f"Summary: {identical}/{total} identical, {diffs} structural diffs, {crashes} crashes")
    return 0 if (identical == total) else 1


if __name__ == '__main__':
    sys.exit(main())
