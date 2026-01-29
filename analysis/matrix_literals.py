# Ethan Doughty
# matrix_literals.py

from __future__ import annotations
from typing import List

from runtime.shapes import Shape, Dim, dims_definitely_conflict, join_dim, sum_dims


def as_matrix_shape(s: Shape) -> Shape:
    """Treat scalar as 1x1 matrix for concatenation."""
    if s.is_scalar():
        return Shape.matrix(1, 1)
    return s


def infer_matrix_literal_shape(
    shape_rows: List[List[Shape]],
    line: int,
    warnings: List[str],
) -> Shape:
    """
    Shared matrix-literal concatenation checker/inferencer.
    Ported from old analysis.py, but operates on already-evaluated Shapes.

    - scalars treated as 1x1 matrices for concat
    - horizontal concat requires equal row counts within each literal row
    - vertical concat requires equal col counts across literal rows
    - definite concat mismatch => warnings + overall Shape.unknown()
    """
    had_definite_error = False

    # Empty literal []
    if len(shape_rows) == 0:
        return Shape.matrix(0, 0)

    row_heights: List[Dim] = []
    row_widths: List[Dim] = []

    for r, row in enumerate(shape_rows):
        # If somehow an empty row exists, treat as unknown (rare/unexpected).
        if len(row) == 0:
            had_definite_error = True
            warnings.append(
                f"Line {line}: Empty row in matrix literal. Treating result as unknown."
            )
            row_heights.append(None)
            row_widths.append(None)
            continue

        elem_rows: List[Dim] = []
        elem_cols: List[Dim] = []

        for s0 in row:
            s = as_matrix_shape(s0)

            if s.is_unknown():
                elem_rows.append(None)
                elem_cols.append(None)
            elif s.is_matrix():
                elem_rows.append(s.rows)
                elem_cols.append(s.cols)
            else:
                elem_rows.append(None)
                elem_cols.append(None)

        # Horizontal concat constraint inside this row
        height = elem_rows[0]
        for rr in elem_rows[1:]:
            if dims_definitely_conflict(height, rr):
                had_definite_error = True
                warnings.append(
                    f"Line {line}: Horizontal concatenation requires equal row counts in row {r+1}; "
                    f"got {height} and {rr} in matrix literal."
                )
            height = join_dim(height, rr)

        width = sum_dims(elem_cols)
        row_heights.append(height)
        row_widths.append(width)

    # Vertical concat constraint across rows
    common_width = row_widths[0]
    for w in row_widths[1:]:
        if dims_definitely_conflict(common_width, w):
            had_definite_error = True
            warnings.append(
                f"Line {line}: Vertical concatenation requires equal column counts across rows; "
                f"got {common_width} and {w} in matrix literal."
            )
        common_width = join_dim(common_width, w)

    total_height = sum_dims(row_heights)

    if had_definite_error:
        return Shape.unknown()

    return Shape.matrix(total_height, common_width)