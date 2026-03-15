#!/usr/bin/env python3
"""Verify computational equivalence between MATLAB (Octave) and transpiled Python.

Runs .m files through GNU Octave and through conformal-migrate + Python,
then compares stdout output with numeric tolerance.

Usage:
    python3 tools/verify_equiv.py [OPTIONS]

Options:
    --binary PATH   Path to conformal-migrate binary
    --dir DIR       Directory of .m test files (default: tests/migrate_equiv/)
    --file FILE     Test a single .m file
    --verbose       Show passing tests and full output
    --tolerance N   Numeric comparison tolerance (default: 1e-10)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def make_env():
    """Build subprocess environment with DOTNET_ROOT."""
    env = os.environ.copy()
    dotnet_root = os.path.expanduser("~/.dotnet")
    if os.path.isdir(dotnet_root):
        env["DOTNET_ROOT"] = dotnet_root
    env.setdefault("TMPDIR", "/tmp/claude-0/dotnet-tmp")
    return env


def run_octave(m_file, timeout=30):
    """Run a .m file through GNU Octave. Returns (ok, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["octave", "--no-gui", "--silent", "--norc", str(m_file)],
            capture_output=True, text=True, timeout=timeout,
        )
        # Filter out Qt/X11 warnings from stderr
        stderr = "\n".join(
            l for l in result.stderr.splitlines()
            if not l.startswith("QStandard") and not l.startswith("warning: ")
        ).strip()
        return result.returncode == 0, result.stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except FileNotFoundError:
        return False, "", "octave not found"


def run_transpiled(m_file, binary, env, timeout=30):
    """Transpile a .m file and run the result through Python. Returns (ok, stdout, stderr)."""
    # Step 1: Transpile
    try:
        migrate = subprocess.run(
            [binary, str(m_file), "--stdout"],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "", "migrate timeout"
    except FileNotFoundError:
        return False, "", "binary not found"

    if migrate.returncode != 0:
        return False, "", f"migrate failed: {migrate.stderr.strip()}"

    py_code = migrate.stdout
    if not py_code.strip():
        return False, "", "empty migration output"

    # Step 2: Run Python
    try:
        result = subprocess.run(
            [sys.executable, "-c", py_code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "python timeout"


def extract_values(text):
    """Extract a flat list of numeric/string values from output.

    Strips numpy brackets, Octave formatting, and extracts raw values.
    This handles the key difference: Octave prints column vectors vertically
    (one value per line) while numpy prints them inline with brackets.

    Returns list of values (floats or strings).
    """
    # Normalize boolean representations
    text = re.sub(r'\bTrue\b', '1', text)
    text = re.sub(r'\bFalse\b', '0', text)

    # Remove Octave special formatting lines
    text = re.sub(r'Diagonal Matrix\s*', '', text)
    text = re.sub(r'Columns \d+ through \d+:\s*', '', text)

    # Strip numpy brackets, parens (tuples), and trailing dots
    text = text.replace('[', ' ').replace(']', ' ')
    text = text.replace('(', ' ').replace(')', ' ')
    text = text.replace(',', ' ')

    # Extract all tokens
    tokens = text.split()
    values = []
    for tok in tokens:
        # Remove trailing dots (e.g., "0." -> "0")
        try:
            values.append(float(tok))
        except ValueError:
            values.append(tok)
    return values


def compare_values(octave_vals, python_vals, tolerance):
    """Compare flat value sequences with numeric tolerance.

    Returns list of (index, octave_val, python_val, reason) for mismatches.
    """
    mismatches = []
    max_len = max(len(octave_vals), len(python_vals))

    if len(octave_vals) != len(python_vals):
        mismatches.append((
            0, f"({len(octave_vals)} values)", f"({len(python_vals)} values)",
            f"value count: {len(octave_vals)} vs {len(python_vals)}"
        ))
        # Still compare what we can
        max_len = min(len(octave_vals), len(python_vals))

    for i in range(max_len):
        ov = octave_vals[i] if i < len(octave_vals) else "<missing>"
        pv = python_vals[i] if i < len(python_vals) else "<missing>"

        if ov == pv:
            continue

        # Both numeric?
        if isinstance(ov, float) and isinstance(pv, float):
            if abs(ov - pv) > tolerance:
                mismatches.append((i + 1, str(ov), str(pv), f"numeric: {ov} vs {pv}"))
        else:
            mismatches.append((i + 1, str(ov), str(pv), f"value: {ov} vs {pv}"))

    return mismatches


def verify_one(m_file, binary, env, tolerance, verbose):
    """Verify a single file. Returns (name, status, details)."""
    name = Path(m_file).stem

    # Run Octave
    oct_ok, oct_out, oct_err = run_octave(m_file)
    if not oct_ok:
        return name, "OCTAVE_FAIL", oct_err

    # Run transpiled Python
    py_ok, py_out, py_err = run_transpiled(m_file, binary, env)
    if not py_ok:
        return name, "PYTHON_FAIL", py_err

    # Compare outputs: extract flat value sequences and compare
    oct_vals = extract_values(oct_out)
    py_vals = extract_values(py_out)

    mismatches = compare_values(oct_vals, py_vals, tolerance)

    if not mismatches:
        return name, "EQUIV", None

    details = []
    for line_num, ol, pl, reason in mismatches:
        details.append(f"  line {line_num}: {reason}")
        details.append(f"    octave: {ol}")
        details.append(f"    python: {pl}")
    return name, "MISMATCH", "\n".join(details)


def main():
    parser = argparse.ArgumentParser(
        description="Verify computational equivalence between Octave and transpiled Python"
    )
    parser.add_argument("--binary", default="src/migrate/bin/Release/net8.0/conformal-migrate")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tolerance", type=float, default=5e-4,
                        help="Numeric tolerance (default 5e-4 to accommodate Octave's 4-digit display)")
    args = parser.parse_args()

    env = make_env()

    # Resolve binary path
    binary = args.binary
    if not os.path.isabs(binary):
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / binary
        if candidate.exists():
            binary = str(candidate)

    if not os.path.exists(binary):
        print(f"Error: binary not found at {binary}", file=sys.stderr)
        sys.exit(1)

    # Collect files
    if args.file:
        m_files = [args.file]
    elif args.dir:
        m_files = sorted(Path(args.dir).glob("*.m"))
    else:
        project_root = Path(__file__).resolve().parent.parent
        equiv_dir = project_root / "tests" / "migrate_equiv"
        if equiv_dir.exists():
            m_files = sorted(equiv_dir.glob("*.m"))
        else:
            print(f"Error: {equiv_dir} not found", file=sys.stderr)
            sys.exit(1)

    if not m_files:
        print("No .m files found.", file=sys.stderr)
        sys.exit(1)

    # Run tests
    equiv = 0
    mismatch = 0
    octave_fail = 0
    python_fail = 0

    for m_file in m_files:
        name, status, details = verify_one(
            str(m_file), binary, env, args.tolerance, args.verbose
        )

        if status == "EQUIV":
            equiv += 1
            if args.verbose:
                print(f"EQUIV:       {name}")
        elif status == "MISMATCH":
            mismatch += 1
            print(f"MISMATCH:    {name}")
            if details:
                print(details)
        elif status == "OCTAVE_FAIL":
            octave_fail += 1
            print(f"OCTAVE_FAIL: {name} -- {details}")
        elif status == "PYTHON_FAIL":
            python_fail += 1
            print(f"PYTHON_FAIL: {name} -- {details}")

    total = len(m_files)
    print()
    print(f"===== Equivalence: {equiv}/{total} match, "
          f"{mismatch} mismatches, "
          f"{octave_fail} octave failures, "
          f"{python_fail} python failures =====")

    sys.exit(0 if mismatch == 0 and python_fail == 0 else 1)


if __name__ == "__main__":
    main()
