# Ethan Doughty
# mmshape.py
"""Command-line interface for Mini-MATLAB shape analyzer."""

import argparse
from pathlib import Path

from frontend.matlab_parser import parse_matlab
from frontend.lower_ir import lower_program
from analysis import analyze_program, analyze_program_ir
from analysis.diagnostics import has_unsupported


def run_file(file_path: str, compare: bool, strict: bool = False, fixpoint: bool = False) -> int:
    """Analyze a single Mini-MATLAB file.

    Args:
        file_path: Path to .m file to analyze
        compare: If True, compare legacy vs IR analyzer outputs
        strict: If True, exit with error if unsupported constructs detected
        fixpoint: If True, use fixed-point iteration for loop analysis

    Returns:
        Exit code (0 for success, 1 for error)
    """
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: file not found: {file_path}")
        return 1

    try:
        src = path.read_text()
        syntax_ast = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {file_path}: {e}")
        return 1

    ir_prog = lower_program(syntax_ast)

    # Default behavior: run IR analysis
    env_ir, warnings_ir = analyze_program_ir(ir_prog, fixpoint=fixpoint)

    if compare:
        env_syntax, warnings_syntax = analyze_program(syntax_ast)
        print("==== Compare: syntax vs IR ====")
        print("syntax env:", env_syntax)
        print("IR env    :", env_ir)

        if env_syntax.bindings != env_ir.bindings:
            print("ENV DIFF!")
            print("syntax warnings:", warnings_syntax)
            print("IR warnings    :", warnings_ir)
            return 1

        if len(warnings_syntax) != len(warnings_ir):
            print(f"WARNING COUNT DIFF: {len(warnings_syntax)} vs {len(warnings_ir)}")
            print("syntax warnings:", warnings_syntax)
            print("IR warnings    :", warnings_ir)
        else:
            print("Warnings match.")
        print()

    print(f"=== Analysis for {file_path} ===")
    if not warnings_ir:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for warning in warnings_ir:
            print("  -", warning)

    print("\nFinal environment:")
    print(env_ir)

    # Strict mode: fail if unsupported constructs detected
    if strict and has_unsupported(warnings_ir):
        print("\nSTRICT MODE: Unsupported constructs detected (W_UNSUPPORTED_*)")
        return 1

    return 0


def run_tests(strict: bool = False, fixpoint: bool = False) -> int:
    """Run the full test suite.

    Args:
        strict: If True, exit with error if unsupported constructs detected
        fixpoint: If True, use fixed-point iteration for loop analysis

    Returns:
        Exit code (0 for all tests passed, 1 otherwise)
    """
    # Import here so normal usage doesn't load test code
    import run_all_tests
    return run_all_tests.main(return_code=True, strict=strict, fixpoint=fixpoint)


def main() -> int:
    """Main entry point for the mmshape CLI tool.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        prog="mmshape",
        description="Mini-MATLAB shape/dimension analyzer"
    )
    parser.add_argument("file", nargs="?", help="MATLAB .m file to analyze")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare legacy syntax analyzer vs IR analyzer"
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run test suite"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if unsupported constructs detected"
    )
    parser.add_argument(
        "--fixpoint",
        action="store_true",
        help="Use fixed-point iteration for loop analysis"
    )
    args = parser.parse_args()

    if args.tests:
        return run_tests(strict=args.strict, fixpoint=args.fixpoint)

    if not args.file:
        parser.print_help()
        return 1

    return run_file(args.file, compare=args.compare, strict=args.strict, fixpoint=args.fixpoint)


if __name__ == "__main__":
    raise SystemExit(main())