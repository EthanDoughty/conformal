// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// SARIF 2.1.0 emitter for CI integration. Writes a full run object
// with tool metadata, a rule per warning code, result locations, and
// per-artifact SHA-256 hashes for DO-178C audit traceability. Links
// the false-negative policy via the analysisScope property so an
// auditor can understand the tool's scope without leaving the file.

module SarifEmitter

open System.IO
open System.Security.Cryptography
open System.Text
open System.Text.Json
open WarningCodes
open Diagnostics

// --- Rule metadata: short descriptions for all 53 W_* codes ---

let private ruleDescription (code: WarningCode) : string =
    match code with
    | W_INNER_DIM_MISMATCH          -> "Matrix inner dimensions do not agree"
    | W_ELEMENTWISE_MISMATCH        -> "Elementwise operation dimension mismatch"
    | W_HORZCAT_ROW_MISMATCH        -> "Horizontal concatenation row count mismatch"
    | W_VERTCAT_COL_MISMATCH        -> "Vertical concatenation column count mismatch"
    | W_RESHAPE_MISMATCH            -> "Reshape element count mismatch"
    | W_MLDIVIDE_DIM_MISMATCH       -> "Left division row count mismatch"
    | W_MATRIX_POWER_NON_SQUARE     -> "Matrix power on non-square matrix"
    | W_CONCAT_TYPE_MISMATCH        -> "Concatenation of non-numeric types"
    | W_ARITHMETIC_TYPE_MISMATCH    -> "Arithmetic on non-numeric operands"
    | W_TRANSPOSE_TYPE_MISMATCH     -> "Transpose of non-numeric operand"
    | W_NEGATE_TYPE_MISMATCH        -> "Negation of non-numeric operand"
    | W_NOT_TYPE_MISMATCH           -> "Logical NOT of non-numeric operand"
    | W_STRING_ARITHMETIC           -> "Arithmetic on string operands"
    | W_TOO_MANY_INDICES            -> "Too many indices for 2D matrix"
    | W_RANGE_NON_SCALAR            -> "Non-scalar range endpoints"
    | W_INVALID_RANGE               -> "Invalid range (end < start)"
    | W_NON_SCALAR_INDEX            -> "Non-scalar index argument"
    | W_INDEX_OUT_OF_BOUNDS         -> "Index out of bounds"
    | W_INDEX_ASSIGN_TYPE_MISMATCH  -> "Indexed assignment to non-indexable type"
    | W_UNKNOWN_FUNCTION            -> "Unrecognized function"
    | W_FUNCTION_ARG_COUNT_MISMATCH -> "Function argument count mismatch"
    | W_PROCEDURE_IN_EXPR           -> "Procedure used in expression context"
    | W_RECURSIVE_FUNCTION          -> "Recursive function call"
    | W_RECURSIVE_LAMBDA            -> "Recursive lambda call"
    | W_MULTI_ASSIGN_NON_CALL       -> "Destructuring requires function call"
    | W_MULTI_ASSIGN_COUNT_MISMATCH -> "Multi-assignment return count mismatch"
    | W_LAMBDA_CALL_APPROXIMATE     -> "Lambda call with limited analysis"
    | W_LAMBDA_ARG_COUNT_MISMATCH   -> "Lambda argument count mismatch"
    | W_STRUCT_FIELD_NOT_FOUND      -> "Struct field not found"
    | W_FIELD_ACCESS_NON_STRUCT     -> "Field access on non-struct"
    | W_CURLY_INDEXING_NON_CELL     -> "Curly-brace indexing on non-cell"
    | W_CELL_ASSIGN_NON_CELL        -> "Cell assignment to non-cell"
    | W_RETURN_OUTSIDE_FUNCTION     -> "Return outside function"
    | W_BREAK_OUTSIDE_LOOP          -> "Break outside loop"
    | W_CONTINUE_OUTSIDE_LOOP       -> "Continue outside loop"
    | W_DIVISION_BY_ZERO            -> "Division by zero"
    | W_POSSIBLY_NEGATIVE_DIM       -> "Non-positive dimension"
    | W_CONSTRAINT_CONFLICT         -> "Dimension constraint conflict"
    | W_SUSPICIOUS_COMPARISON       -> "Suspicious matrix-scalar comparison"
    | W_MATRIX_COMPARISON           -> "Matrix-to-matrix comparison"
    | W_LOGICAL_OP_NON_SCALAR       -> "Logical operator on non-scalar operands"
    | W_REASSIGN_INCOMPATIBLE       -> "Incompatible reassignment"
    | W_UNSUPPORTED_STMT            -> "Unsupported statement"
    | W_EXTERNAL_PARSE_ERROR        -> "External file parse error"
    | W_END_OUTSIDE_INDEXING        -> "End keyword outside indexing"
    | W_MATRIX_LIT_EMPTY_ROW        -> "Empty row in matrix literal"
    | W_CELLFUN_NON_UNIFORM         -> "cellfun with non-uniform output"
    | W_CODER_VARIABLE_SIZE         -> "Variable-size array (MATLAB Coder)"
    | W_CODER_CELL_ARRAY            -> "Cell array (MATLAB Coder)"
    | W_CODER_DYNAMIC_FIELD         -> "Dynamic struct field (MATLAB Coder)"
    | W_CODER_TRY_CATCH             -> "try/catch (MATLAB Coder)"
    | W_CODER_UNSUPPORTED_BUILTIN   -> "Unsupported builtin (MATLAB Coder)"
    | W_CODER_RECURSION             -> "Recursion (MATLAB Coder)"
    | W_STRICT_MODE                 -> "Strict mode violation"

// All codes in stable order (matches WarningCode DU declaration order).
let private allCodes : WarningCode list =
    [ W_INNER_DIM_MISMATCH; W_ELEMENTWISE_MISMATCH; W_HORZCAT_ROW_MISMATCH
      W_VERTCAT_COL_MISMATCH; W_RESHAPE_MISMATCH; W_MLDIVIDE_DIM_MISMATCH
      W_MATRIX_POWER_NON_SQUARE; W_CONCAT_TYPE_MISMATCH
      W_ARITHMETIC_TYPE_MISMATCH; W_TRANSPOSE_TYPE_MISMATCH
      W_NEGATE_TYPE_MISMATCH; W_NOT_TYPE_MISMATCH; W_STRING_ARITHMETIC
      W_TOO_MANY_INDICES; W_RANGE_NON_SCALAR; W_INVALID_RANGE
      W_NON_SCALAR_INDEX; W_INDEX_OUT_OF_BOUNDS; W_INDEX_ASSIGN_TYPE_MISMATCH
      W_UNKNOWN_FUNCTION; W_FUNCTION_ARG_COUNT_MISMATCH
      W_PROCEDURE_IN_EXPR; W_RECURSIVE_FUNCTION; W_RECURSIVE_LAMBDA
      W_MULTI_ASSIGN_NON_CALL; W_MULTI_ASSIGN_COUNT_MISMATCH
      W_LAMBDA_CALL_APPROXIMATE; W_LAMBDA_ARG_COUNT_MISMATCH
      W_STRUCT_FIELD_NOT_FOUND; W_FIELD_ACCESS_NON_STRUCT
      W_CURLY_INDEXING_NON_CELL; W_CELL_ASSIGN_NON_CELL
      W_RETURN_OUTSIDE_FUNCTION; W_BREAK_OUTSIDE_LOOP; W_CONTINUE_OUTSIDE_LOOP
      W_DIVISION_BY_ZERO; W_POSSIBLY_NEGATIVE_DIM; W_CONSTRAINT_CONFLICT
      W_SUSPICIOUS_COMPARISON; W_MATRIX_COMPARISON; W_LOGICAL_OP_NON_SCALAR
      W_REASSIGN_INCOMPATIBLE; W_UNSUPPORTED_STMT; W_EXTERNAL_PARSE_ERROR
      W_END_OUTSIDE_INDEXING; W_MATRIX_LIT_EMPTY_ROW; W_CELLFUN_NON_UNIFORM
      W_CODER_VARIABLE_SIZE; W_CODER_CELL_ARRAY; W_CODER_DYNAMIC_FIELD
      W_CODER_TRY_CATCH; W_CODER_UNSUPPORTED_BUILTIN; W_CODER_RECURSION
      W_STRICT_MODE ]

// Build index map from WarningCode -> 0-based position in allCodes.
let private codeIndexMap : Map<WarningCode, int> =
    allCodes |> List.mapi (fun i c -> (c, i)) |> Map.ofList

// Error-level codes (same set as LspDiagnostics.ERROR_CODES).
let private errorCodes : Set<WarningCode> =
    Set.ofList [
        W_INNER_DIM_MISMATCH; W_ELEMENTWISE_MISMATCH; W_CONSTRAINT_CONFLICT
        W_HORZCAT_ROW_MISMATCH; W_VERTCAT_COL_MISMATCH; W_RESHAPE_MISMATCH
        W_INDEX_OUT_OF_BOUNDS; W_DIVISION_BY_ZERO
        W_ARITHMETIC_TYPE_MISMATCH; W_TRANSPOSE_TYPE_MISMATCH
        W_NEGATE_TYPE_MISMATCH; W_CONCAT_TYPE_MISMATCH
        W_INDEX_ASSIGN_TYPE_MISMATCH; W_POSSIBLY_NEGATIVE_DIM
        W_FUNCTION_ARG_COUNT_MISMATCH; W_LAMBDA_ARG_COUNT_MISMATCH
        W_MULTI_ASSIGN_COUNT_MISMATCH; W_MULTI_ASSIGN_NON_CALL
        W_PROCEDURE_IN_EXPR; W_BREAK_OUTSIDE_LOOP; W_CONTINUE_OUTSIDE_LOOP
        W_STRICT_MODE; W_MLDIVIDE_DIM_MISMATCH; W_MATRIX_POWER_NON_SQUARE
    ]

let private sarifLevel (code: WarningCode) : string =
    match code with
    | W_UNSUPPORTED_STMT -> "note"
    | c when Set.contains c errorCodes -> "error"
    | _ -> "warning"

// --- SARIF 2.1.0 emitter ---

// Emit SARIF 2.1.0 JSON for the given diagnostics to the provided stream.
// Compute SHA-256 hex digest of source text.
let private computeSha256 (source: string) : string =
    let bytes = Encoding.UTF8.GetBytes(source)
    let hash = SHA256.HashData(bytes)
    hash |> Array.map (fun b -> b.ToString("x2")) |> String.concat ""

let emitSarif (stream: Stream) (relativeUri: string) (diagnostics: Diagnostic list) (version: string) (source: string) (coverage: (int * int * int * int) option) : unit =
    let opts = JsonWriterOptions(Indented = true)
    use writer = new Utf8JsonWriter(stream, opts)

    writer.WriteStartObject()
    writer.WriteString("$schema", "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json")
    writer.WriteString("version", "2.1.0")

    // runs array (single run)
    writer.WriteStartArray("runs")
    writer.WriteStartObject()

    // tool.driver
    writer.WriteStartObject("tool")
    writer.WriteStartObject("driver")
    writer.WriteString("name", "Conformal")
    writer.WriteString("version", version)
    writer.WriteString("informationUri", "https://conformaltools.com")

    // rules array
    writer.WriteStartArray("rules")
    for code in allCodes do
        writer.WriteStartObject()
        writer.WriteString("id", codeString code)
        writer.WriteStartObject("shortDescription")
        writer.WriteString("text", ruleDescription code)
        writer.WriteEndObject() // shortDescription
        writer.WriteStartObject("defaultConfiguration")
        writer.WriteString("level", sarifLevel code)
        writer.WriteEndObject() // defaultConfiguration
        writer.WriteString("helpUri", $"https://conformaltools.com/docs/warnings/{codeString code}")
        writer.WriteEndObject() // rule
    writer.WriteEndArray() // rules

    writer.WriteEndObject() // driver
    writer.WriteEndObject() // tool

    // Normalize URI: forward slashes only
    let uri = relativeUri.Replace('\\', '/')

    // artifacts array with file hash for audit traceability
    writer.WriteStartArray("artifacts")
    writer.WriteStartObject()
    writer.WriteStartObject("location")
    writer.WriteString("uri", uri)
    writer.WriteEndObject() // location
    writer.WriteStartObject("hashes")
    writer.WriteString("sha-256", computeSha256 source)
    writer.WriteEndObject() // hashes
    writer.WriteEndObject() // artifact
    writer.WriteEndArray() // artifacts

    // results array
    writer.WriteStartArray("results")
    for d in diagnostics do
        writer.WriteStartObject()
        writer.WriteString("ruleId", codeString d.code)
        match codeIndexMap.TryFind d.code with
        | Some idx -> writer.WriteNumber("ruleIndex", idx)
        | None -> ()
        writer.WriteString("level", sarifLevel d.code)
        writer.WriteStartObject("message")
        writer.WriteString("text", d.message)
        writer.WriteEndObject() // message
        writer.WriteStartArray("locations")
        writer.WriteStartObject()
        writer.WriteStartObject("physicalLocation")
        writer.WriteStartObject("artifactLocation")
        writer.WriteString("uri", uri)
        writer.WriteEndObject() // artifactLocation
        writer.WriteStartObject("region")
        writer.WriteNumber("startLine", d.line)
        if d.col > 0 then
            writer.WriteNumber("startColumn", d.col)
        writer.WriteEndObject() // region
        writer.WriteEndObject() // physicalLocation
        writer.WriteEndObject() // location
        writer.WriteEndArray() // locations

        // relatedLocations: call stack frames
        if not d.callStack.IsEmpty then
            writer.WriteStartArray("relatedLocations")
            for i, (funcName, callLine) in d.callStack |> List.indexed do
                writer.WriteStartObject()
                writer.WriteNumber("id", i)
                writer.WriteStartObject("physicalLocation")
                writer.WriteStartObject("artifactLocation")
                writer.WriteString("uri", uri)
                writer.WriteEndObject() // artifactLocation
                writer.WriteStartObject("region")
                writer.WriteNumber("startLine", callLine)
                writer.WriteEndObject() // region
                writer.WriteEndObject() // physicalLocation
                writer.WriteStartObject("message")
                writer.WriteString("text", $"in {funcName}, called from line {callLine}")
                writer.WriteEndObject() // message
                writer.WriteEndObject() // relatedLocation
            writer.WriteEndArray() // relatedLocations

        writer.WriteEndObject() // result
    writer.WriteEndArray() // results

    // run.properties: analysis scope and shape coverage metrics
    writer.WriteStartObject("properties")
    writer.WriteString("analysisScope", "https://conformaltools.com/docs/false-negative-policy")
    match coverage with
    | Some (tracked, partial, untracked, total) ->
        writer.WriteNumber("shapeCoverage.tracked", tracked)
        writer.WriteNumber("shapeCoverage.partial", partial)
        writer.WriteNumber("shapeCoverage.untracked", untracked)
        writer.WriteNumber("shapeCoverage.total", total)
        if total > 0 then
            writer.WriteNumber("shapeCoverage.rate", System.Math.Round(float tracked / float total, 3))
    | None -> ()
    writer.WriteEndObject() // properties

    writer.WriteEndObject() // run
    writer.WriteEndArray() // runs

    writer.WriteEndObject() // root
    writer.Flush()

// --- Multi-file SARIF emitter for --batch --format sarif ---

/// One analyzed file's contribution to a batch SARIF run.
type BatchFileEntry = {
    relUri: string
    diagnostics: Diagnostic list
    source: string
    coverage: (int * int * int * int) option
}

// Emit a single SARIF 2.1.0 run covering multiple files.
// All files go in artifacts[]; results reference their file by artifactLocation.index.
// run.properties carries aggregate shape coverage.
let emitBatchSarif (stream: Stream) (files: BatchFileEntry list) (version: string) : unit =
    let opts = JsonWriterOptions(Indented = true)
    use writer = new Utf8JsonWriter(stream, opts)

    writer.WriteStartObject()
    writer.WriteString("$schema", "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json")
    writer.WriteString("version", "2.1.0")

    writer.WriteStartArray("runs")
    writer.WriteStartObject()

    // tool.driver (same metadata as single-file)
    writer.WriteStartObject("tool")
    writer.WriteStartObject("driver")
    writer.WriteString("name", "Conformal")
    writer.WriteString("version", version)
    writer.WriteString("informationUri", "https://conformaltools.com")
    writer.WriteStartArray("rules")
    for code in allCodes do
        writer.WriteStartObject()
        writer.WriteString("id", codeString code)
        writer.WriteStartObject("shortDescription")
        writer.WriteString("text", ruleDescription code)
        writer.WriteEndObject()
        writer.WriteStartObject("defaultConfiguration")
        writer.WriteString("level", sarifLevel code)
        writer.WriteEndObject()
        writer.WriteString("helpUri", $"https://conformaltools.com/docs/warnings/{codeString code}")
        writer.WriteEndObject()
    writer.WriteEndArray()
    writer.WriteEndObject() // driver
    writer.WriteEndObject() // tool

    // Build URI -> artifact index map for result references
    let artifactIndex =
        files |> List.mapi (fun i f -> f.relUri.Replace('\\', '/'), i) |> Map.ofList

    // artifacts: one per file with SHA-256 hash
    writer.WriteStartArray("artifacts")
    for f in files do
        let uri = f.relUri.Replace('\\', '/')
        writer.WriteStartObject()
        writer.WriteStartObject("location")
        writer.WriteString("uri", uri)
        writer.WriteEndObject()
        writer.WriteStartObject("hashes")
        writer.WriteString("sha-256", computeSha256 f.source)
        writer.WriteEndObject()
        writer.WriteEndObject()
    writer.WriteEndArray()

    // results: all diagnostics from all files
    writer.WriteStartArray("results")
    for f in files do
        let uri = f.relUri.Replace('\\', '/')
        let idx = Map.find uri artifactIndex
        for d in f.diagnostics do
            writer.WriteStartObject()
            writer.WriteString("ruleId", codeString d.code)
            match codeIndexMap.TryFind d.code with
            | Some ruleIdx -> writer.WriteNumber("ruleIndex", ruleIdx)
            | None -> ()
            writer.WriteString("level", sarifLevel d.code)
            writer.WriteStartObject("message")
            writer.WriteString("text", d.message)
            writer.WriteEndObject()
            writer.WriteStartArray("locations")
            writer.WriteStartObject()
            writer.WriteStartObject("physicalLocation")
            writer.WriteStartObject("artifactLocation")
            writer.WriteString("uri", uri)
            writer.WriteNumber("index", idx)
            writer.WriteEndObject()
            writer.WriteStartObject("region")
            writer.WriteNumber("startLine", d.line)
            if d.col > 0 then
                writer.WriteNumber("startColumn", d.col)
            writer.WriteEndObject()
            writer.WriteEndObject()
            writer.WriteEndObject()
            writer.WriteEndArray()
            if not d.callStack.IsEmpty then
                writer.WriteStartArray("relatedLocations")
                for i, (funcName, callLine) in d.callStack |> List.indexed do
                    writer.WriteStartObject()
                    writer.WriteNumber("id", i)
                    writer.WriteStartObject("physicalLocation")
                    writer.WriteStartObject("artifactLocation")
                    writer.WriteString("uri", uri)
                    writer.WriteNumber("index", idx)
                    writer.WriteEndObject()
                    writer.WriteStartObject("region")
                    writer.WriteNumber("startLine", callLine)
                    writer.WriteEndObject()
                    writer.WriteEndObject()
                    writer.WriteStartObject("message")
                    writer.WriteString("text", $"in {funcName}, called from line {callLine}")
                    writer.WriteEndObject()
                    writer.WriteEndObject()
                writer.WriteEndArray()
            writer.WriteEndObject() // result
    writer.WriteEndArray() // results

    // run.properties: aggregate shape coverage across all files
    writer.WriteStartObject("properties")
    writer.WriteString("analysisScope", "https://conformaltools.com/docs/false-negative-policy")
    let aggTracked   = files |> List.sumBy (fun f -> match f.coverage with Some (t,_,_,_) -> t | None -> 0)
    let aggPartial   = files |> List.sumBy (fun f -> match f.coverage with Some (_,p,_,_) -> p | None -> 0)
    let aggUntracked = files |> List.sumBy (fun f -> match f.coverage with Some (_,_,u,_) -> u | None -> 0)
    let aggTotal     = files |> List.sumBy (fun f -> match f.coverage with Some (_,_,_,t) -> t | None -> 0)
    if aggTotal > 0 then
        writer.WriteNumber("shapeCoverage.tracked", aggTracked)
        writer.WriteNumber("shapeCoverage.partial", aggPartial)
        writer.WriteNumber("shapeCoverage.untracked", aggUntracked)
        writer.WriteNumber("shapeCoverage.total", aggTotal)
        writer.WriteNumber("shapeCoverage.rate", System.Math.Round(float aggTracked / float aggTotal, 3))
    writer.WriteEndObject()

    writer.WriteEndObject() // run
    writer.WriteEndArray() // runs
    writer.WriteEndObject() // root
    writer.Flush()
