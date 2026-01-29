# Ethan Doughty
# mmshape.py
import argparse
import sys
from pathlib import Path

from frontend.matlab_parser import parse_matlab
from frontend.lower_ir import lower_program
from frontend.pipeline import parse_syntax, lower_to_ir
from analysis import analyze_program, analyze_program_ir

def run_file(path: str, compare: bool) -> int:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    try:
        src = p.read_text()
        syntax_ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {path}: {e}")
        return 1

    ir_prog = lower_program(syntax_ast)

    # Default behavior: run IR analysis
    env_ir, warnings_ir = analyze_program_ir(ir_prog)

    if compare:
        env_s, warnings_s = analyze_program(syntax_ast)
        print("==== Compare: syntax vs IR ====")
        print("syntax env:", env_s)
        print("IR env    :", env_ir)

        if env_s.bindings != env_ir.bindings:
            print("ENV DIFF!")
            print("syntax warnings:", warnings_s)
            print("IR warnings    :", warnings_ir)
            return 1

        if len(warnings_s) != len(warnings_ir):
            print(f"WARNING COUNT DIFF: {len(warnings_s)} vs {len(warnings_ir)}")
            print("syntax warnings:", warnings_s)
            print("IR warnings    :", warnings_ir)
        else:
            print("Warnings match.")
        print()

    print(f"=== Analysis for {path} ===")
    if not warnings_ir:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings_ir:
            print("  -", w)

    print("\nFinal environment:")
    print(env_ir)
    return 0


def run_tests() -> int:
    # Import here so normal usage doesn't load test code
    import run_all_tests
    return 0 if run_all_tests.main(return_code=True) == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="mmshape", description="Mini-MATLAB shape/dimension analyzer")
    ap.add_argument("file", nargs="?", help="MATLAB .m file to analyze")
    ap.add_argument("--compare", action="store_true", help="Compare legacy syntax analyzer vs IR analyzer")
    ap.add_argument("--tests", action="store_true", help="Run test suite")
    args = ap.parse_args()

    if args.tests:
        return run_tests()

    if not args.file:
        ap.print_help()
        return 1

    return run_file(args.file, compare=args.compare)


if __name__ == "__main__":
    raise SystemExit(main())