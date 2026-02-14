# Ethan Doughty
# analysis_core.py
"""Core shape analysis utilities and compatibility checks."""

from runtime.shapes import *


def shapes_definitely_incompatible(old: Shape, new: Shape) -> bool:
    """Check if two shapes are provably incompatible for variable reassignment.

    Args:
        old: Previous shape of a variable
        new: New shape being assigned

    Returns:
        True if shapes are definitely incompatible, False otherwise
    """
    # If either is unknown or bottom, don't claim incompatibility
    # (bottom is compatible with everything; unknown is indeterminate)
    if old.is_unknown() or new.is_unknown() or old.is_bottom() or new.is_bottom():
        return False

    # Scalar vs matrix is definitely incompatible for reassignment
    if old.is_scalar() and new.is_matrix():
        return True
    if old.is_matrix() and new.is_scalar():
        return True

    # Matrix vs matrix: check any provable dimension conflicts
    if old.is_matrix() and new.is_matrix():
        if dims_definitely_conflict(old.rows, new.rows):
            return True
        if dims_definitely_conflict(old.cols, new.cols):
            return True

    return False


def elementwise_result_shape(left: Shape, right: Shape) -> Shape:
    """Compute result shape for elementwise operations (+, -, .*, ./).

    Args:
        left: Shape of left operand
        right: Shape of right operand

    Returns:
        Result shape with joined dimensions
    """
    if left.is_unknown() or right.is_unknown():
        return Shape.unknown()
    if left.is_scalar() and right.is_scalar():
        return Shape.scalar()
    if left.is_matrix() and right.is_matrix():
        if dims_definitely_conflict(left.rows, right.rows) or dims_definitely_conflict(left.cols, right.cols):
            return Shape.unknown()
        return Shape.matrix(join_dim(left.rows, right.rows), join_dim(left.cols, right.cols))
    return Shape.unknown()


def elementwise_definitely_mismatch(left: Shape, right: Shape) -> bool:
    """Check if elementwise operation has provable dimension mismatch.

    Args:
        left: Shape of left operand
        right: Shape of right operand

    Returns:
        True if dimensions definitely conflict
    """
    return (
        left.is_matrix() and right.is_matrix()
        and (
            dims_definitely_conflict(left.rows, right.rows)
            or dims_definitely_conflict(left.cols, right.cols)
        )
    )


def matmul_result_shape(left: Shape, right: Shape) -> Shape:
    """Compute result shape for matrix multiplication.

    Args:
        left: Shape of left operand
        right: Shape of right operand

    Returns:
        Result shape (left.rows x right.cols for matrices)
    """
    if left.is_scalar() and right.is_scalar():
        return Shape.scalar()
    if left.is_scalar() and right.is_matrix():
        return right
    if right.is_scalar() and left.is_matrix():
        return left
    if left.is_matrix() and right.is_matrix():
        if dims_definitely_conflict(left.cols, right.rows):
            return Shape.unknown()
        return Shape.matrix(left.rows, right.cols)
    return Shape.unknown()


def matmul_definitely_mismatch(left: Shape, right: Shape) -> bool:
    """Check if matrix multiplication has provable inner dimension mismatch.

    Args:
        left: Shape of left operand
        right: Shape of right operand

    Returns:
        True if left.cols != right.rows provably
    """
    return left.is_matrix() and right.is_matrix() and dims_definitely_conflict(left.cols, right.rows)