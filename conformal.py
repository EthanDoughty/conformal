# Ethan Doughty
# conformal.py
"""Command-line interface for Conformal — MATLAB shape analyzer."""

import argparse
import time
from pathlib import Path

from frontend.matlab_parser import parse_matlab
from analysis import analyze_program_ir, generate_witnesses
from analysis.diagnostics import has_unsupported, STRICT_ONLY_CODES
from analysis.context import AnalysisContext
from analysis.workspace import scan_workspace


def run_file(file_path: str, strict: bool = False, fixpoint: bool = False,
             benchmark: bool = False, witness: str = '') -> int:
    """Analyze a single MATLAB file.

    Args:
        file_path: Path to .m file to analyze
        strict: If True, exit with error if unsupported constructs detected
        fixpoint: If True, use fixed-point iteration for loop analysis
        benchmark: If True, print timing breakdown
        witness: Witness mode: 'enrich', 'filter', 'tag', or '' (disabled)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: file not found: {file_path}")
        return 1

    t_start = time.perf_counter()

    try:
        src = path.read_text(errors='replace')
        t_read = time.perf_counter()
        ir_prog = parse_matlab(src)
    except Exception as e:
        print(f"Error while parsing {file_path}: {e}")
        return 1

    t_parse = time.perf_counter()

    # Scan workspace for external functions
    ext = scan_workspace(path.parent, exclude=path.name)
    t_scan = time.perf_counter()

    ctx = AnalysisContext(fixpoint=fixpoint, external_functions=ext)

    # Run IR analysis
    env, warnings = analyze_program_ir(ir_prog, fixpoint=fixpoint, ctx=ctx)
    t_analyze = time.perf_counter()

    # Filter low-confidence warnings in default mode
    if not strict:
        warnings = [w for w in warnings if w.code not in STRICT_ONLY_CODES]

    # Generate witnesses if requested
    witnesses = {}
    if witness:
        witnesses = generate_witnesses(ctx.conflict_sites)

    # Filter mode: only show warnings with witnesses
    if witness == 'filter':
        warnings = [w for w in warnings if (w.line, w.code) in witnesses]

    print(f"=== Analysis for {file_path} ===")
    if not warnings:
        print("No dimension warnings.")
    else:
        print("Warnings:")
        for w in warnings:
            # Tag mode: prefix with [confirmed] or [possible]
            if witness == 'tag':
                tag = "[confirmed]" if (w.line, w.code) in witnesses else "[possible]"
                print(f"  - {tag} {w}")
            else:
                print("  -", w)
            # Enrich/tag modes: show witness details below
            if witness in ('enrich', 'tag', 'filter'):
                key = (w.line, w.code)
                wt = witnesses.get(key)
                if wt is not None:
                    print(f"    Witness: {wt.explanation}")
                    if wt.path:
                        path_parts = []
                        for desc, taken, ln in wt.path:
                            branch_str = "true branch" if taken else "false branch"
                            path_parts.append(f"line {ln} (if {desc}, {branch_str})")
                        print(f"    Path: {' -> '.join(path_parts)}")

    print("\nFinal environment:")
    print(env)

    if benchmark:
        line_count = src.count('\n') + 1
        total_ms = (t_analyze - t_start) * 1000
        print(f"\n--- Benchmark ({line_count} lines, {len(warnings)} warnings) ---")
        print(f"  Read:      {(t_read - t_start) * 1000:7.1f}ms")
        print(f"  Parse:     {(t_parse - t_read) * 1000:7.1f}ms")
        print(f"  Workspace: {(t_scan - t_parse) * 1000:7.1f}ms")
        print(f"  Analyze:   {(t_analyze - t_scan) * 1000:7.1f}ms")
        print(f"  Total:     {total_ms:7.1f}ms")
        if total_ms > 0:
            print(f"  Throughput: {line_count / total_ms * 1000:.0f} lines/sec")

    # Strict mode: fail if unsupported constructs detected
    if strict and has_unsupported(warnings):
        print("\nSTRICT MODE: Unsupported constructs detected (W_UNSUPPORTED_*)")
        return 1

    return 0


def run_tests(strict: bool = False, fixpoint: bool = False,
              benchmark: bool = False) -> int:
    """Run the full test suite.

    Args:
        strict: If True, exit with error if unsupported constructs detected
        fixpoint: If True, use fixed-point iteration for loop analysis
        benchmark: If True, print timing summary after tests

    Returns:
        Exit code (0 for all tests passed, 1 otherwise)
    """
    import run_all_tests

    if not benchmark:
        return run_all_tests.main(return_code=True, strict=strict, fixpoint=fixpoint)

    t_start = time.perf_counter()
    result = run_all_tests.main(return_code=True, strict=strict, fixpoint=fixpoint)
    t_end = time.perf_counter()

    total_ms = (t_end - t_start) * 1000
    n_tests = len(run_all_tests.TEST_FILES)

    print(f"\n--- Benchmark ---")
    print(f"  Tests:     {n_tests}")
    print(f"  Total:     {total_ms:.0f}ms")
    print(f"  Per-test:  {total_ms / n_tests:.1f}ms avg")
    print(f"  Mode:      {'fixpoint' if fixpoint else 'normal'}")

    return result


def main() -> int:
    """Main entry point for the conformal CLI tool.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        prog="conformal",
        description="Conformal — static shape/dimension analyzer for MATLAB"
    )
    parser.add_argument("file", nargs="?", help="MATLAB .m file to analyze")
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run test suite"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Show all warnings including informational and low-confidence diagnostics"
    )
    parser.add_argument(
        "--fixpoint",
        action="store_true",
        help="Use fixed-point iteration for loop analysis"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print timing breakdown for analysis phases"
    )
    parser.add_argument(
        "--witness",
        nargs="?",
        const="enrich",
        default="",
        metavar="MODE",
        help="Witness mode: enrich (default, show witnesses below warnings), "
             "filter (only show confirmed bugs), tag (prefix [confirmed]/[possible])"
    )
    args = parser.parse_args()

    # Resolve ambiguity: --witness may consume the filename as its optional arg.
    # If witness looks like a file path, shift it to args.file and default witness to 'enrich'.
    witness_mode = args.witness or ''
    if witness_mode and witness_mode not in ('enrich', 'filter', 'tag'):
        if args.file is None:
            args.file = witness_mode
            witness_mode = 'enrich'
        else:
            parser.error(f"--witness mode must be enrich, filter, or tag (got '{witness_mode}')")

    if args.tests:
        return run_tests(strict=args.strict, fixpoint=args.fixpoint,
                         benchmark=args.benchmark)

    if not args.file:
        parser.print_help()
        return 1

    return run_file(args.file, strict=args.strict, fixpoint=args.fixpoint,
                    benchmark=args.benchmark, witness=witness_mode)


if __name__ == "__main__":
    raise SystemExit(main())
