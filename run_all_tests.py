# Ethan Doughty
# run_all_tests.py

import sys
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple

from frontend.matlab_parser import parse_matlab
from analysis import analyze_program_ir
from analysis.context import AnalysisContext
from analysis.workspace import scan_workspace
from analysis.diagnostics import has_unsupported
from runtime.shapes import Shape

_REPO_ROOT = Path(__file__).resolve().parent


def test_sort_key(path: str) -> str:
    """Sort test files alphabetically by full path."""
    return path


TEST_FILES = sorted(glob.glob(str(_REPO_ROOT / "tests/**/*.m"), recursive=True), key=test_sort_key)

EXPECT_RE = re.compile(r"%\s*EXPECT:\s*(.+)$")
EXPECT_FIXPOINT_RE = re.compile(r"%\s*EXPECT_FIXPOINT:\s*(.+)$")
EXPECT_WARNINGS_RE = re.compile(r"warnings\s*=\s*(\d+)\s*$", re.IGNORECASE)
EXPECT_BINDING_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(.+)$")


def normalize_shape_str(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def parse_expectations(src: str, fixpoint: bool = False) -> Tuple[Dict[str, str], int | None]:
    expected_shapes: Dict[str, str] = {}
    expected_warning_count: int | None = None
    # Fixpoint-specific expectations override defaults when in fixpoint mode
    fp_shapes: Dict[str, str] = {}
    fp_warning_count: int | None = None
    has_fp_expectations = False

    for line in src.splitlines():
        stripped = line.strip()

        # Check for EXPECT_FIXPOINT: lines
        m_fp = EXPECT_FIXPOINT_RE.match(stripped)
        if m_fp:
            has_fp_expectations = True
            payload = m_fp.group(1).strip()
            m_warn = EXPECT_WARNINGS_RE.match(payload)
            if m_warn:
                fp_warning_count = int(m_warn.group(1))
                continue
            m_bind = EXPECT_BINDING_RE.match(payload)
            if m_bind:
                fp_shapes[m_bind.group(1)] = normalize_shape_str(m_bind.group(2).strip())
            continue

        # Check for EXPECT: lines
        m = EXPECT_RE.match(stripped)
        if not m:
            continue

        payload = m.group(1).strip()

        m_warn = EXPECT_WARNINGS_RE.match(payload)
        if m_warn:
            expected_warning_count = int(m_warn.group(1))
            continue

        m_bind = EXPECT_BINDING_RE.match(payload)
        if m_bind:
            var = m_bind.group(1)
            shape_str = m_bind.group(2).strip()
            expected_shapes[var] = normalize_shape_str(shape_str)

    # In fixpoint mode, override with fixpoint-specific expectations
    if fixpoint and has_fp_expectations:
        expected_shapes.update(fp_shapes)
        if fp_warning_count is not None:
            expected_warning_count = fp_warning_count

    return expected_shapes, expected_warning_count


def run_test(path: str, fixpoint: bool = False) -> tuple[bool, bool]:
    """Run a test file and return (passed, has_unsupported_warnings).

    Args:
        path: Path to test file
        fixpoint: If True, use fixed-point iteration for loop analysis

    Returns:
        Tuple of (test_passed, has_unsupported_stmt_warnings)
    """
    print(f"===== Analysis for {path}")
    if not os.path.exists(path):
        print("ERROR: file not found\n")
        return False, False

    src = open(path, "r", errors='replace').read()
    expected_shapes, expected_warning_count = parse_expectations(src, fixpoint=fixpoint)

    try:
        ir_prog = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}\n")
        return False, False

    # Scan workspace for external functions
    test_path = Path(path)
    ext = scan_workspace(test_path.parent, exclude=test_path.name)
    ctx = AnalysisContext(fixpoint=fixpoint, external_functions=ext)

    # IR is the source of truth for expectations.
    env, warnings = analyze_program_ir(ir_prog, fixpoint=fixpoint, ctx=ctx)

    # Check for unsupported statement warnings using shared helper
    has_unsupported_warnings = has_unsupported(warnings)

    if not warnings:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings:
            print("-", w)

    print("Final environment:")
    print("   ", env)

    passed = True

    if expected_warning_count is not None and len(warnings) != expected_warning_count:
        print(f"ASSERT FAIL: expected warnings = {expected_warning_count}, got {len(warnings)}")
        passed = False

    for var, expected_shape in expected_shapes.items():
        actual: Shape = env.get(var)
        actual_str = normalize_shape_str(str(actual))
        if actual_str != expected_shape:
            print(f"ASSERT FAIL: expected {var} = {expected_shape}, got {actual_str}")
            passed = False

    print("ASSERTIONS:", "PASS" if passed else "FAIL")
    print()
    return passed, has_unsupported_warnings


def _run_structural_tests() -> tuple[int, int]:
    """Run structural Python tests and return (total, passed) counts."""
    import importlib.util
    structural_dir = _REPO_ROOT / "tests" / "structural"
    test_files = sorted(structural_dir.glob("test_*.py"))
    total = 0
    ok = 0
    for tf in test_files:
        spec = importlib.util.spec_from_file_location(tf.stem, tf)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        test_fns = [
            (name, obj)
            for name, obj in vars(mod).items()
            if name.startswith("test_") and callable(obj)
        ]
        for name, func in test_fns:
            total += 1
            try:
                func()
                print(f"===== Structural: {tf.stem}.{name}: PASS")
                ok += 1
            except AssertionError as e:
                print(f"===== Structural: {tf.stem}.{name}: FAIL: {e}")
            except Exception as e:
                print(f"===== Structural: {tf.stem}.{name}: ERROR: {type(e).__name__}: {e}")
    return total, ok


def main(return_code: bool = False, strict: bool = False, fixpoint: bool = False) -> int:
    """Run all tests.

    Args:
        return_code: If True, return exit code instead of exiting
        strict: If True, exit with error if unsupported constructs detected
        fixpoint: If True, use fixed-point iteration for loop analysis

    Returns:
        Exit code (0 for all tests passed, 1 otherwise)
    """
    total = 0
    ok = 0
    # Accept strict and fixpoint from parameters or sys.argv for backwards compatibility
    strict = strict or "--strict" in sys.argv
    fixpoint = fixpoint or "--fixpoint" in sys.argv
    any_unsupported = False

    for path in TEST_FILES:
        total += 1
        passed, has_unsupported = run_test(path, fixpoint=fixpoint)
        if passed:
            ok += 1
        if has_unsupported:
            any_unsupported = True

    # Run structural Python tests alongside .m tests
    structural_total, structural_ok = _run_structural_tests()
    total += structural_total
    ok += structural_ok

    print(f"===== Summary: {ok}/{total} tests passed =====")

    rc = 0 if ok == total else 1

    # In strict mode, exit with error if any unsupported constructs found
    if strict and any_unsupported:
        print("STRICT MODE: Unsupported constructs detected (W_UNSUPPORTED_*)")
        rc = 1

    if return_code:
        return rc
    return rc

if __name__ == "__main__":
    sys.exit(main())