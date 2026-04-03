module Suppressions

open WarningCodes
open Diagnostics

// ---------------------------------------------------------------------------
// Suppression directives: parse % conformal:disable / disable-next-line
// from raw MATLAB source text, then filter diagnostics post-analysis.
// Pure module (no I/O, no mutation). Fable-compatible.
// ---------------------------------------------------------------------------

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
