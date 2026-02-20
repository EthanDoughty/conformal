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
from runtime.shapes import Shape, ScalarShape, MatrixShape


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


def test_is_empty_matrix():
    """Shape.is_empty_matrix() predicate works correctly."""
    assert not Shape.scalar().is_empty_matrix(), "scalar should not be empty matrix"
    assert not Shape.matrix(3, 4).is_empty_matrix(), "3x4 matrix should not be empty matrix"
    assert not Shape.unknown().is_empty_matrix(), "unknown should not be empty matrix"
    assert Shape.matrix(0, 0).is_empty_matrix(), "0x0 matrix should be empty matrix"
    print("PASS: Shape.is_empty_matrix() predicate is correct")


def test_shape_subclass_hierarchy():
    """Shape factory methods return typed subclasses with correct behavior."""
    # isinstance checks
    assert isinstance(Shape.scalar(), ScalarShape), "Shape.scalar() must return ScalarShape"
    assert isinstance(Shape.matrix(3, 4), MatrixShape), "Shape.matrix() must return MatrixShape"

    # kind attribute (class attribute on each subclass)
    assert Shape.scalar().kind == "scalar", "ScalarShape.kind must be 'scalar'"
    assert Shape.matrix(3, 4).kind == "matrix", "MatrixShape.kind must be 'matrix'"

    # Structural equality
    assert Shape.scalar() == Shape.scalar(), "ScalarShape() == ScalarShape() must be True"
    assert Shape.matrix(3, 4) == Shape.matrix(3, 4), "MatrixShape(3,4) == MatrixShape(3,4) must be True"

    # Hash stability
    assert hash(Shape.scalar()) == hash(Shape.scalar()), "hash(ScalarShape()) must be stable"
    assert hash(Shape.matrix(3, 4)) == hash(Shape.matrix(3, 4)), "hash(MatrixShape(3,4)) must be stable"

    # __str__ output unchanged
    assert str(Shape.matrix(3, 4)) == "matrix[3 x 4]", f"str mismatch: {str(Shape.matrix(3, 4))!r}"
    assert str(Shape.scalar()) == "scalar", f"str mismatch: {str(Shape.scalar())!r}"
    assert str(Shape.unknown()) == "unknown", f"str mismatch: {str(Shape.unknown())!r}"

    # Cross-kind inequality
    assert Shape.scalar() != Shape.unknown(), "ScalarShape != UnknownShape must be True"
    assert Shape.scalar() != Shape.matrix(3, 4), "ScalarShape != MatrixShape must be True"

    # Predicates
    assert Shape.scalar().is_scalar(), "ScalarShape.is_scalar() must be True"
    assert not Shape.scalar().is_matrix(), "ScalarShape.is_matrix() must be False"

    print("PASS: Shape subclass hierarchy is correct")


if __name__ == '__main__':
    test_lower_expr_raises_on_unknown_tag()
    test_lower_stmt_returns_opaque_on_unknown_tag()
    test_lower_stmt_skip_unchanged()
    test_is_empty_matrix()
    test_shape_subclass_hierarchy()
    print("\nAll tests passed.")
