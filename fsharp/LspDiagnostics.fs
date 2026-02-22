module LspDiagnostics

open Ionide.LanguageServerProtocol.Types
open Diagnostics
open Witness

// ---------------------------------------------------------------------------
// Codes that represent definite errors (will crash at runtime).
// Port of lsp/diagnostics.py ERROR_CODES set.
// ---------------------------------------------------------------------------

let ERROR_CODES : Set<string> =
    Set.ofList [
        "W_INNER_DIM_MISMATCH"
        "W_ELEMENTWISE_MISMATCH"
        "W_CONSTRAINT_CONFLICT"
        "W_HORZCAT_ROW_MISMATCH"
        "W_VERTCAT_COL_MISMATCH"
        "W_RESHAPE_MISMATCH"
        "W_INDEX_OUT_OF_BOUNDS"
        "W_DIVISION_BY_ZERO"
        "W_ARITHMETIC_TYPE_MISMATCH"
        "W_TRANSPOSE_TYPE_MISMATCH"
        "W_NEGATE_TYPE_MISMATCH"
        "W_CONCAT_TYPE_MISMATCH"
        "W_INDEX_ASSIGN_TYPE_MISMATCH"
        "W_POSSIBLY_NEGATIVE_DIM"
        "W_FUNCTION_ARG_COUNT_MISMATCH"
        "W_LAMBDA_ARG_COUNT_MISMATCH"
        "W_MULTI_ASSIGN_COUNT_MISMATCH"
        "W_MULTI_ASSIGN_NON_CALL"
        "W_MULTI_ASSIGN_BUILTIN"
        "W_PROCEDURE_IN_EXPR"
        "W_BREAK_OUTSIDE_LOOP"
        "W_CONTINUE_OUTSIDE_LOOP"
        "W_STRICT_MODE"
        "W_MLDIVIDE_DIM_MISMATCH"
        "W_MATRIX_POWER_NON_SQUARE"
    ]

// ---------------------------------------------------------------------------
// toLspDiagnostic: convert a Conformal Diagnostic to an LSP Diagnostic.
// Port of lsp/diagnostics.py to_lsp_diagnostic().
// ---------------------------------------------------------------------------

let toLspDiagnostic
    (d: Diagnostics.Diagnostic)
    (sourceLines: string array)
    (uri: string)
    (witness: Witness option)
    : Ionide.LanguageServerProtocol.Types.Diagnostic =

    // Convert 1-based Conformal line to 0-based LSP line
    let lineNum = d.line - 1

    // End character: full line length if in range, else 0
    let endChar =
        if lineNum >= 0 && lineNum < sourceLines.Length then
            sourceLines.[lineNum].Length
        else 0

    // Start column: 1-based col -> 0-based LSP col
    let startChar = if d.col > 0 then d.col - 1 else 0

    let range : Range = {
        Start = { Line = uint32 lineNum; Character = uint32 startChar }
        End   = { Line = uint32 lineNum; Character = uint32 endChar }
    }

    // Severity mapping
    let severity =
        if Set.contains d.code ERROR_CODES then DiagnosticSeverity.Error
        elif d.code.StartsWith("W_UNSUPPORTED_") then DiagnosticSeverity.Hint
        else DiagnosticSeverity.Warning

    // Tags: Unnecessary for unsupported constructs
    let tags : DiagnosticTag[] option =
        if d.code.StartsWith("W_UNSUPPORTED_") then
            Some [| DiagnosticTag.Unnecessary |]
        else None

    // Related information for related_line
    let relatedInformation : DiagnosticRelatedInformation[] option =
        match d.relatedLine with
        | None -> None
        | Some relLine ->
            let relLineNum = relLine - 1
            let relEndChar =
                if relLineNum >= 0 && relLineNum < sourceLines.Length then
                    sourceLines.[relLineNum].Length
                else 0
            let relRange : Range = {
                Start = { Line = uint32 relLineNum; Character = 0u }
                End   = { Line = uint32 relLineNum; Character = uint32 relEndChar }
            }
            let relLoc : Location = { Uri = uri; Range = relRange }
            let relInfo : DiagnosticRelatedInformation = {
                Location = relLoc
                Message  = "Related: see line " + string relLine
            }
            Some [| relInfo |]

    // Enrich message with witness if available
    let message =
        match witness with
        | None -> d.message
        | Some w ->
            let baseMsg = d.message + "\nWitness: " + w.explanation
            if w.path.IsEmpty then baseMsg
            else
                let parts =
                    w.path
                    |> List.map (fun (desc, taken, ln) ->
                        let branchStr = if taken then "true branch" else "false branch"
                        "line " + string ln + " (if " + desc + ", " + branchStr + ")")
                    |> String.concat " -> "
                baseMsg + "\nPath: " + parts

    {
        Range              = range
        Severity           = Some severity
        Code               = Some (U2.C2 d.code)
        Source             = Some "conformal"
        Message            = message
        Tags               = tags
        RelatedInformation = relatedInformation
        CodeDescription    = None
        Data               = None
    }
