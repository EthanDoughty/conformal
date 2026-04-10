// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Parses % conformal:disable and disable-next-line directives directly
// from MATLAB source text, then filters the diagnostic list after
// analysis. Pure module with no I/O or mutation, so it runs identically
// under .NET and under Fable in the VS Code extension.

module Suppressions

open WarningCodes
open Diagnostics

type SuppressionInfo = {
    fileCodes: Set<string>
    lineCodes: Map<int, Set<string>>
}

let empty : SuppressionInfo = { fileCodes = Set.empty; lineCodes = Map.empty }

/// Extract W_* codes from the text after the directive keyword.
/// E.g. "W_UNKNOWN_FUNCTION W_STRUCT_FIELD_NOT_FOUND" -> ["W_UNKNOWN_FUNCTION"; "W_STRUCT_FIELD_NOT_FOUND"]
let private extractCodes (text: string) : string list =
    text.Split(' ')
    |> Array.toList
    |> List.map (fun s -> s.Trim())
    |> List.filter (fun s -> s.StartsWith("W_") && s.Length > 2)

/// Parse suppression directives from raw source text.
let parseSuppressions (source: string) : SuppressionInfo =
    let lines = source.Split('\n')
    let mutable fileCodes = Set.empty<string>
    let mutable lineCodes = Map.empty<int, Set<string>>
    let mutable seenNonComment = false

    let addLineCodes (lineNum: int) (codes: string list) =
        if not codes.IsEmpty then
            let existing =
                match Map.tryFind lineNum lineCodes with
                | Some s -> s
                | None -> Set.empty
            let merged = codes |> List.fold (fun acc c -> Set.add c acc) existing
            lineCodes <- Map.add lineNum merged lineCodes

    for i in 0 .. lines.Length - 1 do
        let line = lines.[i]
        let trimmed = line.Trim()
        let lineNum = i + 1  // 1-based

        // Track whether we've passed the file-level comment header
        if not seenNonComment then
            if trimmed <> "" && not (trimmed.StartsWith("%")) then
                seenNonComment <- true

        // Look for the directive in the line.
        // We search for "conformal:disable-next-line" first (longer match),
        // then "conformal:disable".
        let nextLineMarker = "conformal:disable-next-line"
        let disableMarker = "conformal:disable"

        // Find the % that starts the comment containing the directive
        let percentIdx = line.IndexOf('%')
        if percentIdx >= 0 then
            let commentPart = line.Substring(percentIdx + 1)

            // Check for disable-next-line first
            let nlIdx = commentPart.IndexOf(nextLineMarker)
            if nlIdx >= 0 then
                let afterDirective = commentPart.Substring(nlIdx + nextLineMarker.Length)
                let codes = extractCodes afterDirective
                // Applies to the NEXT line
                addLineCodes (lineNum + 1) codes
            else
                // Check for plain disable
                let dIdx = commentPart.IndexOf(disableMarker)
                if dIdx >= 0 then
                    let afterDirective = commentPart.Substring(dIdx + disableMarker.Length)
                    let codes = extractCodes afterDirective

                    if not codes.IsEmpty then
                        // Determine if this is file-level or inline
                        let beforePercent = line.Substring(0, percentIdx).Trim()
                        let isOwnLine = beforePercent = ""
                        let isFileLevel = isOwnLine && not seenNonComment

                        if isFileLevel then
                            fileCodes <- codes |> List.fold (fun acc c -> Set.add c acc) fileCodes
                        else
                            // Inline: suppress on this line only
                            addLineCodes lineNum codes

    { fileCodes = fileCodes; lineCodes = lineCodes }

/// Check if a diagnostic should be suppressed.
let isSuppressed (info: SuppressionInfo) (line: int) (code: string) : bool =
    Set.contains code info.fileCodes ||
    match Map.tryFind line info.lineCodes with
    | Some codes -> Set.contains code codes
    | None -> false

/// Filter a list of diagnostics, removing those suppressed by directives.
let filterDiagnostics (info: SuppressionInfo) (diagnostics: Diagnostic list) : Diagnostic list =
    diagnostics |> List.filter (fun d ->
        not (isSuppressed info d.line (codeString d.code)))

// ---------------------------------------------------------------------------
// Type annotation directives: parse % conformal:type varname shape
// from raw MATLAB source text (file header only).
// Pure module (no I/O, no mutation). Fable-compatible.
// ---------------------------------------------------------------------------

/// Parse a shape spec token: "[NxM]", "scalar", or "string".
/// Returns Some shape on success, None if unrecognised.
let private parseShapeSpec (spec: string) : Shapes.Shape option =
    let s = spec.Trim()
    if s = "scalar" then
        Some Shapes.Scalar
    elif s = "string" then
        Some Shapes.StringShape
    elif s.StartsWith("[") && s.EndsWith("]") then
        // Strip brackets and split on 'x'
        let inner = s.Substring(1, s.Length - 2)
        let parts = inner.Split('x')
        if parts.Length = 2 then
            match System.Int32.TryParse(parts.[0].Trim()), System.Int32.TryParse(parts.[1].Trim()) with
            | (true, r), (true, c) ->
                Some (Shapes.Matrix(Shapes.Concrete r, Shapes.Concrete c))
            | _ -> None
        else None
    else None

/// Parse % conformal:type directives from the file header (lines before any
/// non-comment code). Returns a map from variable name to seeded shape.
let parseTypeAnnotations (source: string) : Map<string, Shapes.Shape> =
    let lines = source.Split('\n')
    let mutable result = Map.empty<string, Shapes.Shape>
    let mutable seenNonComment = false
    let typeMarker = "conformal:type"

    for i in 0 .. lines.Length - 1 do
        let line = lines.[i]
        let trimmed = line.Trim()

        // Stop collecting annotations once non-comment code appears
        if not seenNonComment then
            if trimmed <> "" && not (trimmed.StartsWith("%")) then
                seenNonComment <- true

        if not seenNonComment then
            let percentIdx = line.IndexOf('%')
            if percentIdx >= 0 then
                let commentPart = line.Substring(percentIdx + 1)
                let tIdx = commentPart.IndexOf(typeMarker)
                if tIdx >= 0 then
                    let afterDirective = commentPart.Substring(tIdx + typeMarker.Length).Trim()
                    // afterDirective should be "varname shapespec"
                    // Split on first whitespace to get varname and shapespec
                    let spaceIdx = afterDirective.IndexOfAny([|' '; '\t'|])
                    if spaceIdx > 0 then
                        let varName  = afterDirective.Substring(0, spaceIdx).Trim()
                        let specStr  = afterDirective.Substring(spaceIdx + 1).Trim()
                        if varName.Length > 0 && specStr.Length > 0 then
                            match parseShapeSpec specStr with
                            | Some shape -> result <- Map.add varName shape result
                            | None -> ()

    result
