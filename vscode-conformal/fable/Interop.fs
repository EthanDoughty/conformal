module Interop

open Ir
open Shapes
open Context
open Diagnostics
open Analysis

// ---------------------------------------------------------------------------
// Serializable result types for TypeScript consumption.
// Plain records with primitive fields — Fable compiles these to JS objects.
// ---------------------------------------------------------------------------

type SerializedDiagnostic = {
    line:        int
    col:         int
    code:        string
    message:     string
    relatedLine: int option
    relatedCol:  int option
}

type FunctionSymbol = {
    name:   string
    line:   int
    parms:  string array
    outputs: string array
}

type AnalysisResult = {
    diagnostics: SerializedDiagnostic array
    env:         (string * string) array   // (varName, shapeString)
    symbols:     FunctionSymbol array
    parseError:  string option
}

// ---------------------------------------------------------------------------
// Workspace injection: the TS host provides file content, not System.IO.
// ---------------------------------------------------------------------------

/// Register external function signatures from pre-read files.
/// The TS host reads sibling .m files and passes (filename, content) pairs.
let private buildExternalMap
    (externalFiles: (string * string) array)
    : Map<string, ExternalSignature> =

    let result = System.Collections.Generic.Dictionary<string, ExternalSignature>()
    for (fileName, content) in externalFiles do
        let key = if fileName.EndsWith(".m") then fileName.[..fileName.Length-3] else fileName
        match Workspace.extractFunctionSignature content with
        | None -> ()
        | Some (funcName, paramCount, returnCount) ->
            result.[key] <- {
                filename    = funcName
                paramCount  = paramCount
                returnCount = returnCount
                sourcePath  = ""  // No filesystem path in browser/Fable context
                body        = None
                parmNames   = []
                outputNames = []
            }
    result |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq


/// Parse MATLAB source and try to load a function body for cross-file analysis.
let private tryParseExternalBody (source: string) : (FunctionSignature * Map<string, FunctionSignature>) option =
    try
        let program = Parser.parseMATLAB source
        let funcDefs =
            program.body
            |> List.choose (fun stmt ->
                match stmt with
                | FunctionDef(_, _, name, parms, outputVars, body) ->
                    Some { name = name; parms = parms; outputVars = outputVars; body = body }
                | _ -> None)
        match funcDefs with
        | [] -> None
        | primary :: rest ->
            let subfunctions = rest |> List.map (fun s -> (s.name, s)) |> Map.ofList
            Some (primary, subfunctions)
    with _ -> None

// ---------------------------------------------------------------------------
// Main entry point: called from TypeScript server.
// ---------------------------------------------------------------------------

/// Analyze MATLAB source code. Returns a plain JS-friendly result object.
let analyzeSource
    (source: string)
    (fixpoint: bool)
    (strict: bool)
    (externalFiles: (string * string) array)
    : AnalysisResult =

    // Parse
    let irProgOpt =
        try Some (Parser.parseMATLAB source)
        with
        | Parser.ParseError msg -> None
        | Lexer.LexError msg -> None
        | _ -> None

    match irProgOpt with
    | None ->
        // Return parse error — TS server will create a diagnostic for it
        let errMsg =
            try
                Parser.parseMATLAB source |> ignore
                "Unknown parse error"
            with
            | Parser.ParseError msg -> msg
            | Lexer.LexError msg -> msg
            | ex -> ex.Message
        { diagnostics = [||]; env = [||]; symbols = [||]; parseError = Some errMsg }

    | Some irProg ->

    // Extract symbols (function definitions) from IR
    let symbols =
        irProg.body
        |> List.choose (fun stmt ->
            match stmt with
            | FunctionDef(line, _, name, parms, outputVars, _) ->
                Some { name = name; line = line; parms = Array.ofList parms; outputs = Array.ofList outputVars }
            | _ -> None)
        |> Array.ofList

    // Set up workspace context
    let extMap = buildExternalMap externalFiles

    let ctx = AnalysisContext()
    ctx.call.fixpoint <- fixpoint
    ctx.cst.strictMode <- strict
    for kv in extMap do
        ctx.ws.externalFunctions.[kv.Key] <- kv.Value

    // Run analysis
    let (env, warnings) =
        try analyzeProgramIr irProg ctx
        with ex ->
            // Analysis crash — return what we have
            (Env.Env.create (), [])

    // Filter strict-only codes in default mode
    let displayWarnings =
        if strict then warnings
        else warnings |> List.filter (fun w -> not (Set.contains w.code STRICT_ONLY_CODES))

    // Serialize diagnostics
    let diags =
        displayWarnings
        |> List.map (fun w ->
            { line = w.line; col = w.col; code = w.code; message = w.message
              relatedLine = w.relatedLine; relatedCol = w.relatedCol })
        |> Array.ofList

    // Serialize environment
    let envPairs =
        env.bindings
        |> Map.toArray
        |> Array.map (fun (k, v) -> (k, shapeToString v))

    { diagnostics = diags; env = envPairs; symbols = symbols; parseError = None }
