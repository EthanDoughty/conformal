"""Structural completeness test for the Shape subclass hierarchy.

This test fails if a new Shape subclass is added without being handled in
join_shape, widen_shape, and related critical functions.

Run directly:   python3 tests/structural/test_shape_completeness.py
Run via pytest: python3 -m pytest tests/structural/test_shape_completeness.py -v
"""

import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.shapes import (
    Shape,
    ScalarShape,
    MatrixShape,
    StringShape,
    StructShape,
    FunctionHandleShape,
    CellShape,
    UnknownShape,
    BottomShape,
    join_shape,
    widen_shape,
)

# ---------------------------------------------------------------------------
# Sample instances: one per subclass
# ---------------------------------------------------------------------------

SAMPLES = [
    ScalarShape(),
    MatrixShape(rows=2, cols=3),
    StringShape(),
    StructShape(),
    FunctionHandleShape(),
    CellShape(rows=1, cols=1),
    UnknownShape(),
    BottomShape(),
]

EXPECTED_SUBCLASS_COUNT = 8


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_subclass_count():
    """Fail fast when a 9th Shape subclass is added without updating this test."""
    subclasses = Shape.__subclasses__()
    assert len(subclasses) == EXPECTED_SUBCLASS_COUNT, (
        f"Expected {EXPECTED_SUBCLASS_COUNT} Shape subclasses, found {len(subclasses)}: "
        f"{[c.__name__ for c in subclasses]}. "
        f"Update EXPECTED_SUBCLASS_COUNT and SAMPLES in this file, then verify "
        f"join_shape and widen_shape handle the new subclass."
    )


def test_join_shape_completeness():
    """join_shape(a, b) must not raise for any pair of Shape instances."""
    for a in SAMPLES:
        for b in SAMPLES:
            result = join_shape(a, b)
            assert isinstance(result, Shape), (
                f"join_shape({type(a).__name__}, {type(b).__name__}) returned "
                f"{type(result).__name__}, expected a Shape"
            )


def test_widen_shape_completeness():
    """widen_shape(a, b) must not raise for any pair of Shape instances."""
    for a in SAMPLES:
        for b in SAMPLES:
            result = widen_shape(a, b)
            assert isinstance(result, Shape), (
                f"widen_shape({type(a).__name__}, {type(b).__name__}) returned "
                f"{type(result).__name__}, expected a Shape"
            )


def test_bottom_is_join_identity():
    """join_shape(s, bottom) == s and join_shape(bottom, s) == s for all s."""
    bot = Shape.bottom()
    for s in SAMPLES:
        assert join_shape(s, bot) == s, (
            f"join_shape({type(s).__name__}, bottom) should equal {s}, "
            f"got {join_shape(s, bot)}"
        )
        assert join_shape(bot, s) == s, (
            f"join_shape(bottom, {type(s).__name__}) should equal {s}, "
            f"got {join_shape(bot, s)}"
        )


def test_unknown_is_join_absorbing():
    """join_shape(s, unknown) is unknown for all non-bottom s."""
    unk = Shape.unknown()
    for s in SAMPLES:
        if s.is_bottom():
            # join(bottom, unknown) = unknown (bottom is identity, so result is unknown)
            assert join_shape(s, unk).is_unknown(), (
                f"join_shape(bottom, unknown) should be unknown"
            )
            assert join_shape(unk, s).is_unknown(), (
                f"join_shape(unknown, bottom) should be unknown"
            )
        else:
            assert join_shape(s, unk).is_unknown(), (
                f"join_shape({type(s).__name__}, unknown) should be unknown, "
                f"got {join_shape(s, unk)}"
            )
            assert join_shape(unk, s).is_unknown(), (
                f"join_shape(unknown, {type(s).__name__}) should be unknown, "
                f"got {join_shape(unk, s)}"
            )


def test_join_idempotent_same_kind():
    """join_shape(s, s) == s for same-kind, same-value pairs."""
    for s in SAMPLES:
        result = join_shape(s, s)
        assert result == s, (
            f"join_shape({type(s).__name__}, {type(s).__name__}) should be idempotent: "
            f"expected {s}, got {result}"
        )


def test_cross_kind_join_is_unknown():
    """join_shape of two non-bottom, non-unknown, different-kind shapes is unknown."""
    # Non-trivial kinds (exclude bottom and unknown from the cross-kind check)
    concrete = [s for s in SAMPLES if not s.is_bottom() and not s.is_unknown()]
    for a in concrete:
        for b in concrete:
            if type(a) is type(b):
                continue
            result = join_shape(a, b)
            assert result.is_unknown(), (
                f"join_shape({type(a).__name__}, {type(b).__name__}) should be unknown "
                f"(different kinds), got {result}"
            )


def test_cross_kind_widen_is_unknown():
    """widen_shape of two non-bottom, non-unknown, different-kind shapes is unknown."""
    concrete = [s for s in SAMPLES if not s.is_bottom() and not s.is_unknown()]
    for a in concrete:
        for b in concrete:
            if type(a) is type(b):
                continue
            result = widen_shape(a, b)
            assert result.is_unknown(), (
                f"widen_shape({type(a).__name__}, {type(b).__name__}) should be unknown "
                f"(different kinds), got {result}"
            )


def test_is_numeric_coverage():
    """is_numeric() returns True for exactly {ScalarShape, MatrixShape, StringShape}."""
    expected_numeric = {ScalarShape, MatrixShape, StringShape}
    for s in SAMPLES:
        if type(s) in expected_numeric:
            assert s.is_numeric(), (
                f"{type(s).__name__}.is_numeric() should return True"
            )
        else:
            assert not s.is_numeric(), (
                f"{type(s).__name__}.is_numeric() should return False"
            )


def test_predicate_coverage():
    """Each Shape subclass has exactly one is_*() predicate returning True."""
    predicates = [
        "is_scalar",
        "is_matrix",
        "is_string",
        "is_struct",
        "is_function_handle",
        "is_cell",
        "is_unknown",
        "is_bottom",
    ]
    for s in SAMPLES:
        true_preds = [p for p in predicates if getattr(s, p)()]
        assert len(true_preds) == 1, (
            f"{type(s).__name__}: expected exactly 1 True predicate among is_*(), "
            f"got {true_preds}"
        )


# ---------------------------------------------------------------------------
# Self-runnable entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    test_fns = [
        test_subclass_count,
        test_join_shape_completeness,
        test_widen_shape_completeness,
        test_bottom_is_join_identity,
        test_unknown_is_join_absorbing,
        test_join_idempotent_same_kind,
        test_cross_kind_join_is_unknown,
        test_cross_kind_widen_is_unknown,
        test_is_numeric_coverage,
        test_predicate_coverage,
    ]
    for func in test_fns:
        try:
            func()
            print(f"  PASS: {func.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {func.__name__}: {e}")
            failures += 1

    print()
    total = len(test_fns)
    ok = total - failures
    print(f"Structural tests: {ok}/{total} passed")
    sys.exit(0 if failures == 0 else 1)
