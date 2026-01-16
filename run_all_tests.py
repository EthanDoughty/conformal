# Ethan Doughty
# run_all_tests.py
import os
import re
from typing import Dict, List, Tuple

from matlab_parser import parse_matlab
from analysis import analyze_program
from shapes import Shape

TEST_FILES = [f"tests/test{i}.m" for i in range(1, 19)]

# Expectations syntax in .m files:
#  % EXPECT: warnings = 0
#  % EXPECT: A = matrix[2 x 3]
#  % EXPECT: x = scalar
#
# shape strings are matched loosely (whitespace-insensitive).
# if an EXPECT is missing, it is not checked.
EXPECT_RE = re.compile(r"%\s*EXPECT:\s*(.+)$")
EXPECT_WARNINGS_RE = re.compile(r"warnings\s*=\s*(\d+)\s*$", re.IGNORECASE)
EXPECT_BINDING_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(.+)$")


def normalize_shape_str(s: str) -> str:
    """Normalize for comparison: remove extra whitespace."""
    return re.sub(r"\s+", "", s.strip())


def parse_expectations(src: str) -> Tuple[Dict[str, str], int | None]:
    """
    Returns (expected_shapes, expected_warning_count).
    expected_shapes maps var -> normalized shape string.
    """
    expected_shapes: Dict[str, str] = {}
    expected_warning_count: int | None = None

    for line in src.splitlines():
        m = EXPECT_RE.match(line.strip())
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
            continue

        # Unrecognized EXPECT line — ignore (future-proofing)
    return expected_shapes, expected_warning_count


def run_test(path: str) -> bool:
    print(f"===== Analysis for {path}")
    if not os.path.exists(path):
        print(f"ERROR: file not found\n")
        return False

    with open(path, "r") as f:
        src = f.read()

    expected_shapes, expected_warning_count = parse_expectations(src)

    try:
        ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}\n")
        return False

    env, warnings = analyze_program(ast)

    # Print diagnostics (same as before)
    if not warnings:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings:
            print("-", w)

    print("Final environment:")
    print("   ", env)

    # Assertions
    passed = True

    if expected_warning_count is not None:
        if len(warnings) != expected_warning_count:
            print(
                f"ASSERT FAIL: expected warnings = {expected_warning_count}, "
                f"got {len(warnings)}"
            )
            passed = False

    for var, expected_shape in expected_shapes.items():
        actual: Shape = env.get(var)
        actual_str = normalize_shape_str(str(actual))
        if actual_str != expected_shape:
            print(
                f"ASSERT FAIL: expected {var} = {expected_shape}, "
                f"got {actual_str}"
            )
            passed = False

    if passed:
        print("ASSERTIONS: PASS")
    else:
        print("ASSERTIONS: FAIL")

    print()
    return passed


def main():
    total = 0
    ok = 0

    for path in TEST_FILES:
        total += 1
        if run_test(path):
            ok += 1

    print(f"===== Summary: {ok}/{total} tests passed =====")


if __name__ == "__main__":
    main()