# Ethan Doughty
# matrix_literals.py

from __future__ import annotations
from typing import List, TYPE_CHECKING

from runtime.shapes import Shape, Dim, dims_definitely_conflict, join_dim, sum_dims
from analysis.constraints import record_constraint

if TYPE_CHECKING:
    from analysis.diagnostics import Diagnostic


def as_matrix_shape(s: Shape) -> Shape:
    """Treat scalar and string as 1x1 matrix for concatenation.

    Strings are treated as scalar-like elements for concatenation purposes.
    Defensive: bottom should never reach here (converted at Var eval boundary),
    but if it does, return unknown.
    """
    if s.is_scalar() or s.is_string():
        return Shape.matrix(1, 1)
    if s.is_bottom():
        # Safety net: bottom leaked into concat logic (should never happen)
        return Shape.unknown()
    return s


def infer_matrix_literal_shape(
    shape_rows: List[List[Shape]],
    line: int,
    warnings: List['Diagnostic'],
    ctx,
    env
) -> Shape:
    """
    Shared matrix-literal concatenation checker/inferencer.
    Ported from old analysis.py, but operates on already-evaluated Shapes.

    - scalars and strings treated as 1x1 matrices for concat
    - if all elements are strings, result is string (horzcat of strings)
    - horizontal concat requires equal row counts within each literal row
    - vertical concat requires equal col counts across literal rows
    - definite concat mismatch => warnings + overall Shape.unknown()
    """
    had_definite_error = False

    # Empty literal []
    if len(shape_rows) == 0:
        return Shape.matrix(0, 0)

    # Check if all elements are strings
    all_strings = True
    for row in shape_rows:
        for elem in row:
            if not elem.is_string():
                all_strings = False
                break
        if not all_strings:
            break

    if all_strings and any(len(row) > 0 for row in shape_rows):
        # All elements are strings: horzcat produces string
        return Shape.string()

    row_heights: List[Dim] = []
    row_widths: List[Dim] = []

    # Track first element kind across entire literal for same-kind concat allowance
    literal_first_kind = None
    if shape_rows and shape_rows[0] and len(shape_rows[0]) > 0:
        literal_first_kind = shape_rows[0][0].kind

    for r, row in enumerate(shape_rows):
        # If somehow an empty row exists, treat as unknown (rare/unexpected).
        if len(row) == 0:
            had_definite_error = True
            from analysis.diagnostics import Diagnostic
            warnings.append(
                Diagnostic(
                    line=line,
                    code="W_MATRIX_LIT_EMPTY_ROW",
                    message="Empty row in matrix literal. Treating result as unknown."
                )
            )
            row_heights.append(None)
            row_widths.append(None)
            continue

        elem_rows: List[Dim] = []
        elem_cols: List[Dim] = []

        for s0 in row:
            # Type check: warn on mixed kinds if either is non-numeric (but not unknown)
            # Skip check if either element is unknown (can't prove type error)
            # Note: scalar/string are treated as 1x1 matrices for concat, so they're compatible with matrix
            if literal_first_kind is not None and literal_first_kind != "unknown" and s0.kind != literal_first_kind and not s0.is_unknown():
                # Check if kinds are incompatible for concat (non-numeric mixing with anything)
                first_is_numeric = literal_first_kind in ("scalar", "matrix", "string")
                current_is_numeric = s0.is_numeric()
                # Only warn if at least one is non-numeric
                if not first_is_numeric or not current_is_numeric:
                    had_definite_error = True
                    import analysis.diagnostics as diag
                    warnings.append(diag.warn_concat_type_mismatch(line, s0))

            s = as_matrix_shape(s0)

            # Empty matrix [] is identity for concatenation — skip it
            if s.is_matrix() and s.rows == 0 and s.cols == 0:
                continue

            if s.is_unknown():
                elem_rows.append(None)
                elem_cols.append(None)
            elif s.is_matrix():
                elem_rows.append(s.rows)
                elem_cols.append(s.cols)
            else:
                elem_rows.append(None)
                elem_cols.append(None)

        # If all elements were empty matrices, skip row entirely (identity for concat)
        if not elem_rows:
            continue

        # Horizontal concat constraint inside this row
        height = elem_rows[0]
        for rr in elem_rows[1:]:
            # Record constraint between consecutive element row dimensions
            record_constraint(ctx, env, height, rr, line)

            if dims_definitely_conflict(height, rr):
                had_definite_error = True
                from analysis.diagnostics import Diagnostic
                from analysis.witness import ConflictSite
                ctx.cst.conflict_sites.append(ConflictSite(
                    dim_a=height, dim_b=rr,
                    line=line, warning_code="W_HORZCAT_ROW_MISMATCH",
                    constraints_snapshot=frozenset(ctx.cst.constraints),
                    scalar_bindings_snapshot=tuple(sorted(ctx.cst.scalar_bindings.items())),
                    value_ranges_snapshot=tuple(sorted(
                        (k, (v.lo, v.hi)) for k, v in ctx.cst.value_ranges.items()
                    )),
                    path_snapshot=tuple(ctx.cst.path_constraints.snapshot()),
                ))
                warnings.append(
                    Diagnostic(
                        line=line,
                        code="W_HORZCAT_ROW_MISMATCH",
                        message=(
                            f"Horizontal concatenation requires equal row counts in row {r+1}; "
                            f"got {height} and {rr} in matrix literal."
                        )
                    )
                )
            height = join_dim(height, rr)

        width = sum_dims(elem_cols)
        row_heights.append(height)
        row_widths.append(width)

    # If all rows were empty matrices, result is empty
    if not row_widths:
        return Shape.matrix(0, 0)

    # Vertical concat constraint across rows
    common_width = row_widths[0]
    for w in row_widths[1:]:
        # Record constraint between consecutive row widths
        record_constraint(ctx, env, common_width, w, line)

        if dims_definitely_conflict(common_width, w):
            had_definite_error = True
            from analysis.diagnostics import Diagnostic
            from analysis.witness import ConflictSite
            ctx.cst.conflict_sites.append(ConflictSite(
                dim_a=common_width, dim_b=w,
                line=line, warning_code="W_VERTCAT_COL_MISMATCH",
                constraints_snapshot=frozenset(ctx.cst.constraints),
                scalar_bindings_snapshot=tuple(sorted(ctx.cst.scalar_bindings.items())),
                value_ranges_snapshot=tuple(sorted(
                    (k, (v.lo, v.hi)) for k, v in ctx.cst.value_ranges.items()
                )),
                path_snapshot=tuple(ctx.cst.path_constraints.snapshot()),
            ))
            warnings.append(
                Diagnostic(
                    line=line,
                    code="W_VERTCAT_COL_MISMATCH",
                    message=(
                        f"Vertical concatenation requires equal column counts across rows; "
                        f"got {common_width} and {w} in matrix literal."
                    )
                )
            )
        common_width = join_dim(common_width, w)

    total_height = sum_dims(row_heights)

    if had_definite_error:
        return Shape.unknown()

    return Shape.matrix(total_height, common_width)