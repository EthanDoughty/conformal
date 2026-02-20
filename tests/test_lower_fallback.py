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


if __name__ == '__main__':
    test_lower_expr_raises_on_unknown_tag()
    test_lower_stmt_returns_opaque_on_unknown_tag()
    test_lower_stmt_skip_unchanged()
    print("\nAll tests passed.")
