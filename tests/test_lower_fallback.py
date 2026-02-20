"""Unit tests for lower_ir.py fallback behavior.

These are Python tests (not .m test files) because lowering failures come from
code bugs, not user MATLAB input. Run with:

    python3 tests/test_lower_fallback.py
"""
import sys
import os

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.lower_ir import lower_expr, lower_stmt
from ir.ir import OpaqueStmt
from runtime.shapes import Shape, ShapeKind


def test_lower_expr_raises_on_unknown_tag():
    """lower_expr must raise ValueError for any unrecognized tag."""
    try:
        lower_expr(['bogus_tag', 1, 2])
        assert False, "Expected ValueError but no exception was raised"
    except ValueError as e:
        assert 'bogus_tag' in str(e), f"ValueError message should mention the tag, got: {e}"
    print("PASS: lower_expr raises ValueError on unknown tag")


def test_lower_stmt_returns_opaque_on_unknown_tag():
    """lower_stmt must return OpaqueStmt (not ExprStmt(Const(0.0))) for unknown tags."""
    import io
    from contextlib import redirect_stderr

    buf = io.StringIO()
    with redirect_stderr(buf):
        result = lower_stmt(['bogus_tag', 42])

    assert isinstance(result, OpaqueStmt), (
        f"Expected OpaqueStmt, got {type(result).__name__}: {result!r}"
    )
    assert result.line == 42, f"Expected line=42, got line={result.line}"
    assert 'bogus_tag' in result.raw, f"Expected tag in raw field, got: {result.raw!r}"

    stderr_output = buf.getvalue()
    assert 'bogus_tag' in stderr_output, (
        f"Expected warning on stderr mentioning the tag, got: {stderr_output!r}"
    )
    print("PASS: lower_stmt returns OpaqueStmt on unknown tag")


def test_lower_stmt_skip_unchanged():
    """The 'skip' handler must still return ExprStmt(Const(0.0)) (intentional no-op)."""
    from ir.ir import ExprStmt, Const

    result = lower_stmt(['skip'])
    assert isinstance(result, ExprStmt), (
        f"Expected ExprStmt for 'skip', got {type(result).__name__}"
    )
    assert isinstance(result.expr, Const), (
        f"Expected Const body for 'skip' ExprStmt, got {type(result.expr).__name__}"
    )
    print("PASS: lower_stmt 'skip' handler is unchanged")


def test_shapekind_str_mixin_backward_compat():
    """ShapeKind.SCALAR == "scalar" must be True due to str mixin."""
    assert ShapeKind.SCALAR == "scalar", "str mixin backward compat broken"
    assert ShapeKind.MATRIX == "matrix", "str mixin backward compat broken"
    assert ShapeKind.STRING == "string", "str mixin backward compat broken"
    assert ShapeKind.STRUCT == "struct", "str mixin backward compat broken"
    assert ShapeKind.FUNCTION_HANDLE == "function_handle", "str mixin backward compat broken"
    assert ShapeKind.CELL == "cell", "str mixin backward compat broken"
    assert ShapeKind.UNKNOWN == "unknown", "str mixin backward compat broken"
    assert ShapeKind.BOTTOM == "bottom", "str mixin backward compat broken"
    print("PASS: ShapeKind str mixin backward compatibility works")


def test_shape_kind_string_coercion():
    """Shape(kind='scalar') still works via __post_init__ coercion."""
    s = Shape(kind="scalar")
    assert s.kind == ShapeKind.SCALAR, "Coercion from string 'scalar' failed"
    assert isinstance(s.kind, ShapeKind), "kind should be ShapeKind enum, not bare str"
    print("PASS: Shape(kind='scalar') coerces to ShapeKind.SCALAR")


def test_shape_kind_typo_raises():
    """Shape(kind='matirx') must raise ValueError (typo detection)."""
    try:
        Shape(kind="matirx")
        assert False, "Expected ValueError but no exception was raised"
    except ValueError:
        pass
    print("PASS: Shape(kind='matirx') raises ValueError")


def test_is_empty_matrix():
    """Shape.is_empty_matrix() predicate works correctly."""
    assert not Shape.scalar().is_empty_matrix(), "scalar should not be empty matrix"
    assert not Shape.matrix(3, 4).is_empty_matrix(), "3x4 matrix should not be empty matrix"
    assert not Shape.unknown().is_empty_matrix(), "unknown should not be empty matrix"
    assert Shape.matrix(0, 0).is_empty_matrix(), "0x0 matrix should be empty matrix"
    print("PASS: Shape.is_empty_matrix() predicate is correct")


if __name__ == '__main__':
    test_lower_expr_raises_on_unknown_tag()
    test_lower_stmt_returns_opaque_on_unknown_tag()
    test_lower_stmt_skip_unchanged()
    test_shapekind_str_mixin_backward_compat()
    test_shape_kind_string_coercion()
    test_shape_kind_typo_raises()
    test_is_empty_matrix()
    print("\nAll tests passed.")
