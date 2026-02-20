"""Unit tests for parser IR emission and shape subsystem.

These are Python tests (not .m test files) because they test internal
code paths. Run with:

    python3 tests/test_lower_fallback.py
"""
import sys
import os

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.matlab_parser import parse_matlab, extract_targets_from_tokens
from ir.ir import Program, For, BinOp, OpaqueStmt, ExprStmt, Const
from runtime.shapes import Shape, ScalarShape, MatrixShape


def test_parse_returns_program():
    """parse_matlab must return a Program IR node, not a list."""
    result = parse_matlab('x = 1;')
    assert isinstance(result, Program), (
        f"Expected Program, got {type(result).__name__}: {result!r}"
    )
    assert len(result.body) == 1, f"Expected 1 statement, got {len(result.body)}"
    print("PASS: parse_matlab returns Program")


def test_extract_targets_from_tokens():
    """extract_targets_from_tokens must correctly identify assigned variables."""
    from frontend.lexer import lex

    # Simple assignment: x = 1
    tokens = lex("x = 1")
    # Filter out EOF
    tokens = [t for t in tokens if t.kind != "EOF" and t.kind != "NEWLINE"]
    result = extract_targets_from_tokens(tokens)
    assert result == ["x"], f"Expected ['x'], got {result!r}"

    # Empty token list
    result = extract_targets_from_tokens([])
    assert result == [], f"Expected [], got {result!r}"

    print("PASS: extract_targets_from_tokens works correctly")


def test_parse_for_simplification():
    """for i = 1:10 must produce For(var='i', it=BinOp(op=':'))."""
    result = parse_matlab('for i = 1:10\n  x = i;\nend\n')
    assert isinstance(result, Program), f"Expected Program, got {type(result)}"
    assert len(result.body) >= 1, "Expected at least 1 statement"
    for_node = result.body[0]
    assert isinstance(for_node, For), f"Expected For node, got {type(for_node).__name__}"
    assert for_node.var == "i", f"Expected var='i', got var={for_node.var!r}"
    assert isinstance(for_node.it, BinOp), (
        f"Expected BinOp iterator, got {type(for_node.it).__name__}"
    )
    assert for_node.it.op == ":", f"Expected op=':', got op={for_node.it.op!r}"
    print("PASS: parse_for simplification produces For(var='i', it=BinOp(op=':'))")


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
    test_parse_returns_program()
    test_extract_targets_from_tokens()
    test_parse_for_simplification()
    test_is_empty_matrix()
    test_shape_subclass_hierarchy()
    print("\nAll tests passed.")
