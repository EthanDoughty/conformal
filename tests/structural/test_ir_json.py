"""Structural round-trip tests for frontend/ir_json.py.

Validates that ir_to_json -> ir_from_json is a lossless round-trip for every
IR node type produced by the Python parser.

Run directly:   python3 tests/structural/test_ir_json.py
Run via runner: python3 conformal.py --tests
"""

import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
from frontend.matlab_parser import parse_matlab
from frontend.ir_json import ir_to_json, ir_from_json, _EXPR_SERIALIZERS, _STMT_SERIALIZERS, _INDEX_ARG_SERIALIZERS


_REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Representative files: at least one per IR node category
# ---------------------------------------------------------------------------

# (label, relative path, notes on node types exercised)
_REPRESENTATIVE = [
    ("basics_mismatch",    "tests/basics/inner_dim_mismatch.m",         "Assign, Var, Const, Apply"),
    ("matrix_literal",     "tests/literals/matrix_lit.m",               "MatrixLit"),
    ("control_flow_if",    "tests/control_flow/elseif_chain.m",         "IfChain"),
    ("loops",              "tests/loops/matrix_growth.m",               "For, While, BinOp"),
    ("functions",          "tests/functions/function_def.m",            "FunctionDef, Return"),
    ("lambda",             "tests/functions/lambda_basic.m",            "Lambda, FuncHandle"),
    ("structs",            "tests/structs/struct_create_assign.m",      "StructAssign"),
    ("cells",              "tests/cells/cell_basic.m",                  "CellLit, CurlyApply, CellAssign"),
    ("stress",             "tests/stress/stress_test_control_flow.m",   "Switch, Try, Break, Continue"),
    ("indexing",           "tests/indexing/index_slice.m",              "Range, Colon, IndexExpr, End"),
]


def _round_trip(rel_path: str):
    """Parse, serialize, deserialize, and compare for one file."""
    path = _REPO_ROOT / rel_path
    src = path.read_text(encoding="utf-8", errors="replace")
    prog = parse_matlab(src)
    json_str = ir_to_json(prog)
    restored = ir_from_json(json_str)
    assert prog == restored, (
        f"Round-trip mismatch for {rel_path}.\n"
        f"Original: {prog}\n"
        f"Restored: {restored}"
    )


# ---------------------------------------------------------------------------
# Per-file round-trip tests
# ---------------------------------------------------------------------------

def test_round_trip_basics_mismatch():
    _round_trip("tests/basics/inner_dim_mismatch.m")


def test_round_trip_matrix_literal():
    _round_trip("tests/literals/matrix_literal.m")


def test_round_trip_if_chain():
    _round_trip("tests/control_flow/elseif_chain.m")


def test_round_trip_loops():
    _round_trip("tests/loops/matrix_growth.m")


def test_round_trip_functions():
    _round_trip("tests/functions/endless_basic.m")


def test_round_trip_lambda():
    _round_trip("tests/functions/lambda_basic.m")


def test_round_trip_structs():
    _round_trip("tests/structs/struct_create_assign.m")


def test_round_trip_cells():
    _round_trip("tests/cells/cell_builtin.m")


def test_round_trip_stress_control_flow():
    _round_trip("tests/stress/stress_test_control_flow.m")


def test_round_trip_indexing():
    _round_trip("tests/indexing/colon_in_matrix_index.m")


# ---------------------------------------------------------------------------
# Completeness: every known type has a serializer entry
# ---------------------------------------------------------------------------

def test_expr_serializer_completeness():
    """Every Expr subclass in ir.ir has an entry in _EXPR_SERIALIZERS."""
    import ir.ir as ir_module
    import inspect

    expr_subclasses = set()
    for name, obj in vars(ir_module).items():
        if (inspect.isclass(obj)
                and issubclass(obj, ir_module.Expr)
                and obj is not ir_module.Expr
                and obj is not ir_module.IndexArg):
            expr_subclasses.add(name)

    missing = expr_subclasses - set(_EXPR_SERIALIZERS.keys())
    assert not missing, (
        f"Expr subclasses missing from _EXPR_SERIALIZERS: {sorted(missing)}"
    )


def test_stmt_serializer_completeness():
    """Every Stmt subclass in ir.ir has an entry in _STMT_SERIALIZERS."""
    import ir.ir as ir_module
    import inspect

    stmt_subclasses = set()
    for name, obj in vars(ir_module).items():
        if (inspect.isclass(obj)
                and issubclass(obj, ir_module.Stmt)
                and obj is not ir_module.Stmt):
            stmt_subclasses.add(name)

    missing = stmt_subclasses - set(_STMT_SERIALIZERS.keys())
    assert not missing, (
        f"Stmt subclasses missing from _STMT_SERIALIZERS: {sorted(missing)}"
    )


def test_index_arg_serializer_completeness():
    """Every IndexArg subclass in ir.ir has an entry in _INDEX_ARG_SERIALIZERS."""
    import ir.ir as ir_module
    import inspect

    indexarg_subclasses = set()
    for name, obj in vars(ir_module).items():
        if (inspect.isclass(obj)
                and issubclass(obj, ir_module.IndexArg)
                and obj is not ir_module.IndexArg):
            indexarg_subclasses.add(name)

    missing = indexarg_subclasses - set(_INDEX_ARG_SERIALIZERS.keys())
    assert not missing, (
        f"IndexArg subclasses missing from _INDEX_ARG_SERIALIZERS: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_program():
    """An empty source file produces Program(body=[]) which round-trips."""
    prog = parse_matlab("")
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored


def test_json_has_type_field():
    """Every serialized node has a 'type' discriminator field."""
    import json
    src = "x = 1 + 2;"
    prog = parse_matlab(src)
    data = json.loads(ir_to_json(prog))
    assert data["type"] == "Program"
    assert data["body"][0]["type"] == "Assign"
    assert data["body"][0]["expr"]["type"] == "BinOp"


def test_switch_with_otherwise():
    """Switch statement with otherwise clause round-trips correctly."""
    src = """
switch x
  case 1
    y = 1;
  case 2
    y = 2;
  otherwise
    y = 0;
end
"""
    prog = parse_matlab(src)
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored
    # Verify the Switch node was found
    switch_stmts = [s for s in prog.body if type(s).__name__ == "Switch"]
    assert switch_stmts, "Expected Switch node in parsed output"
    assert len(switch_stmts[0].cases) == 2
    assert len(switch_stmts[0].otherwise) > 0


def test_nested_lambda():
    """Nested lambda expressions round-trip correctly."""
    src = "f = @(x) @(y) x + y;"
    prog = parse_matlab(src)
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored


def test_matrix_and_cell_literals():
    """MatrixLit and CellLit round-trip with multi-row content."""
    src = """
A = [1 2 3; 4 5 6];
C = {1, 'hello'; 2, [3 4]};
"""
    prog = parse_matlab(src)
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored


def test_index_range_and_colon():
    """Range and Colon IndexArg nodes round-trip correctly."""
    src = """
n = 5;
A = zeros(n, n);
B = A(1:end, :);
"""
    prog = parse_matlab(src)
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored


def test_opaque_stmt_raw_field():
    """OpaqueStmt.raw field is preserved in round-trip."""
    import json
    src = "global x y;"
    prog = parse_matlab(src)
    data = json.loads(ir_to_json(prog))
    # Find the OpaqueStmt
    opaque_nodes = [s for s in data["body"] if s["type"] == "OpaqueStmt"]
    assert opaque_nodes, "Expected at least one OpaqueStmt"
    assert "raw" in opaque_nodes[0], "OpaqueStmt must have 'raw' field"
    restored = ir_from_json(ir_to_json(prog))
    assert prog == restored


# ---------------------------------------------------------------------------
# Comprehensive round-trip on all test files
# ---------------------------------------------------------------------------

def test_round_trip_all_files():
    """Every .m test file round-trips through JSON without loss."""
    test_files = sorted((_REPO_ROOT / "tests").glob("**/*.m"))
    assert len(test_files) > 0, "No .m test files found"
    for f in test_files:
        src = f.read_text(encoding="utf-8", errors="replace")
        prog = parse_matlab(src)
        json_str = ir_to_json(prog)
        restored = ir_from_json(json_str)
        assert prog == restored, f"Round-trip failed for {f.relative_to(_REPO_ROOT)}"


# ---------------------------------------------------------------------------
# Self-runnable entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_fns = [
        test_round_trip_basics_mismatch,
        test_round_trip_matrix_literal,
        test_round_trip_if_chain,
        test_round_trip_loops,
        test_round_trip_functions,
        test_round_trip_lambda,
        test_round_trip_structs,
        test_round_trip_cells,
        test_round_trip_stress_control_flow,
        test_round_trip_indexing,
        test_expr_serializer_completeness,
        test_stmt_serializer_completeness,
        test_index_arg_serializer_completeness,
        test_empty_program,
        test_json_has_type_field,
        test_switch_with_otherwise,
        test_nested_lambda,
        test_matrix_and_cell_literals,
        test_index_range_and_colon,
        test_opaque_stmt_raw_field,
        test_round_trip_all_files,
    ]
    failures = 0
    for func in test_fns:
        try:
            func()
            print(f"  PASS: {func.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {func.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {func.__name__}: {type(e).__name__}: {e}")
            failures += 1

    print()
    total = len(test_fns)
    ok = total - failures
    print(f"IR JSON tests: {ok}/{total} passed")
    sys.exit(0 if failures == 0 else 1)
