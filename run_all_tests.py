# Ethan Doughty
# run_all_tests.py

import sys
import os
import re
from typing import Dict, List, Tuple

from frontend.matlab_parser import parse_matlab
from frontend.lower_ir import lower_program
from analysis import analyze_program_ir, analyze_program_legacy
from runtime.shapes import Shape

TEST_FILES = [f"tests/test{i}.m" for i in range(1, 22)]
COMPARE = False

EXPECT_RE = re.compile(r"%\s*EXPECT:\s*(.+)$")
EXPECT_WARNINGS_RE = re.compile(r"warnings\s*=\s*(\d+)\s*$", re.IGNORECASE)
EXPECT_BINDING_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(.+)$")


def normalize_shape_str(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def parse_expectations(src: str) -> Tuple[Dict[str, str], int | None]:
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

    return expected_shapes, expected_warning_count


def run_test(path: str, compare: bool = True) -> bool:
    print(f"===== Analysis for {path}")
    if not os.path.exists(path):
        print("ERROR: file not found\n")
        return False

    src = open(path, "r").read()
    expected_shapes, expected_warning_count = parse_expectations(src)

    try:
        syntax_ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}\n")
        return False

    ir_prog = lower_program(syntax_ast)

    # IR is the source of truth for expectations.
    env, warnings = analyze_program_ir(ir_prog)

    # Optional: compare legacy vs IR for sanity.
    if COMPARE:
        env_s, warnings_s = analyze_program_legacy(syntax_ast)

        print("==== Compare: syntax vs IR ====")
        print("syntax env:", env_s)
        print("IR env    :", env)

        if env_s.bindings != env.bindings:
            print("ENV DIFF!")
            print("syntax warnings:", warnings_s)
            print("IR warnings    :", warnings)
            print()

        if len(warnings_s) != len(warnings):
            print(f"WARNING COUNT DIFF: {len(warnings_s)} vs {len(warnings)}")
            print("syntax warnings:", warnings_s)
            print("IR warnings    :", warnings)
        else:
            print("Warnings match.")
        print()

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
    return passed


def main(return_code: bool = False) -> int | None:
    total = 0
    ok = 0
    compare = "--compare" in sys.argv

    for path in TEST_FILES:
        total += 1
        if run_test(path, compare=compare):
            ok += 1

    print(f"===== Summary: {ok}/{total} tests passed =====")

    rc = 0 if ok == total else 1
    if return_code:
        return rc
    
if __name__ == "__main__":
    main()