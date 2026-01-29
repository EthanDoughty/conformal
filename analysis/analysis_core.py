# Ethan Doughty
# analysis_core.py
from runtime.shapes import *
from typing import Any, List


def shapes_definitely_incompatible(old: Shape, new: Shape) -> bool:
    # If either is unknown, don't claim incompatibility
    if old.is_unknown() or new.is_unknown():
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

def as_matrix_shape(s: Shape) -> Shape:
    if s.is_scalar():
        return Shape.matrix(1, 1)
    return s

def elementwise_result_shape(left: Shape, right: Shape) -> Shape:
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
    return (
        left.is_matrix() and right.is_matrix()
        and (
            dims_definitely_conflict(left.rows, right.rows)
            or dims_definitely_conflict(left.cols, right.cols)
        )
    )

def matmul_result_shape(left: Shape, right: Shape) -> Shape:
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
    return left.is_matrix() and right.is_matrix() and dims_definitely_conflict(left.cols, right.rows)