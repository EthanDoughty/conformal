module MatrixLiterals

open Shapes
open Env
open Context

// ---------------------------------------------------------------------------
// Matrix/cell literal shape inference.
// Port of analysis/matrix_literals.py
// ---------------------------------------------------------------------------

/// asMatrixShape: treat scalar and string as 1x1 matrix for concatenation.
let asMatrixShape (s: Shape) : Shape =
    match s with
    | Scalar | StringShape -> Matrix(Concrete 1, Concrete 1)
    | Bottom -> UnknownShape
    | _ -> s


/// inferMatrixLiteralShape: shared matrix-literal concatenation checker/inferencer.
/// Operates on already-evaluated shapes.
///
/// evalExprFn callback is needed only for the matrix literal shape inference logic;
/// here we take already-evaluated shape_rows directly.
let inferMatrixLiteralShape
    (shapeRows: Shape list list)
    (line: int)
    (warnings: ResizeArray<Diagnostics.Diagnostic>)
    (ctx: AnalysisContext)
    (env: Env)
    : Shape =

    // Empty literal []
    if shapeRows.IsEmpty then
        Matrix(Concrete 0, Concrete 0)
    else
        // Check if all elements are strings
        let allStrings =
            shapeRows |> List.forall (fun row -> row |> List.forall isString)

        if allStrings && shapeRows |> List.exists (fun row -> not row.IsEmpty) then
            StringShape
        else
            let mutable hadDefiniteError = false

            // Track first element kind across entire literal for same-kind concat allowance
            let literalFirstKind =
                match shapeRows with
                | (firstElem :: _) :: _ -> Some firstElem
                | _ -> None

            let rowHeights = System.Collections.Generic.List<Dim>()
            let rowWidths  = System.Collections.Generic.List<Dim>()

            for r, row in shapeRows |> List.mapi (fun i row -> (i, row)) do
                // Empty row (rare/unexpected)
                if row.IsEmpty then
                    hadDefiniteError <- true
                    warnings.Add(Diagnostics.makeDiag line "W_MATRIX_LIT_EMPTY_ROW"
                        "Empty row in matrix literal. Treating result as unknown.")
                    rowHeights.Add(Unknown)
                    rowWidths.Add(Unknown)
                else
                    let elemRows = System.Collections.Generic.List<Dim>()
                    let elemCols = System.Collections.Generic.List<Dim>()

                    for s0 in row do
                        // Type check: warn on mixed kinds if non-numeric mixing
                        match literalFirstKind with
                        | Some firstShape when not (isUnknown firstShape) && s0 <> firstShape && not (isUnknown s0) ->
                            let firstIsNumeric = isNumeric firstShape
                            let currentIsNumeric = isNumeric s0
                            if not firstIsNumeric || not currentIsNumeric then
                                hadDefiniteError <- true
                                warnings.Add(Diagnostics.warnConcatTypeMismatch line s0)
                        | _ -> ()

                        let s = asMatrixShape s0

                        // Empty matrix [] is identity for concatenation — skip it
                        if not (isEmptyMatrix s) then
                            match s with
                            | UnknownShape ->
                                elemRows.Add(Unknown)
                                elemCols.Add(Unknown)
                            | Matrix(sr, sc) ->
                                elemRows.Add(sr)
                                elemCols.Add(sc)
                            | _ ->
                                elemRows.Add(Unknown)
                                elemCols.Add(Unknown)

                    // If all elements were empty matrices, skip row entirely
                    if elemRows.Count > 0 then
                        // Horizontal concat constraint inside this row
                        let mutable height = elemRows.[0]
                        for i in 1 .. elemRows.Count - 1 do
                            let rr = elemRows.[i]
                            Constraints.recordConstraint ctx env height rr line
                            let height' = Constraints.resolveDim ctx height
                            let rr'    = Constraints.resolveDim ctx rr

                            if dimsDefinitelyConflict height' rr' then
                                hadDefiniteError <- true
                                warnings.Add(Diagnostics.makeDiag line "W_HORZCAT_ROW_MISMATCH"
                                    ("Horizontal concatenation requires equal row counts in row " + string (r + 1) +
                                     "; got " + dimStr height + " and " + dimStr rr + " in matrix literal."))
                            height <- joinDim height' rr'

                        let width = sumDims (elemCols |> Seq.toList)
                        rowHeights.Add(height)
                        rowWidths.Add(width)

            // If all rows were empty matrices, result is empty
            if rowWidths.Count = 0 then
                Matrix(Concrete 0, Concrete 0)
            else
                // Vertical concat constraint across rows
                let mutable commonWidth = rowWidths.[0]
                for i in 1 .. rowWidths.Count - 1 do
                    let w = rowWidths.[i]
                    Constraints.recordConstraint ctx env commonWidth w line
                    let commonWidth' = Constraints.resolveDim ctx commonWidth
                    let w'           = Constraints.resolveDim ctx w

                    if dimsDefinitelyConflict commonWidth' w' then
                        hadDefiniteError <- true
                        warnings.Add(Diagnostics.makeDiag line "W_VERTCAT_COL_MISMATCH"
                            ("Vertical concatenation requires equal column counts across rows; got " +
                             dimStr commonWidth + " and " + dimStr w + " in matrix literal."))
                    commonWidth <- joinDim commonWidth' w'

                let totalHeight = sumDims (rowHeights |> Seq.toList)

                if hadDefiniteError then UnknownShape
                else Matrix(totalHeight, commonWidth)
