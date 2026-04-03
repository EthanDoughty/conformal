module Interop

open Ir
open Shapes
open Context
open WarningCodes
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
    name:    string
    line:    int
    col:     int
    parms:   string array
    outputs: string array
}

type AssignmentHint = {
    name:  string
    line:  int
    col:   int
    shape: string
}

type SerializedParseError = {
    message:   string
    startLine: int
    startCol:  int
    endLine:   int
    endCol:    int
}

type AnalysisResult = {
    diagnostics:       SerializedDiagnostic array
    env:               (string * string) array   // (varName, shapeString)
    symbols:           FunctionSymbol array
    parseError:        string option
    parseErrorLine:    int option
    parseErrorCol:     int option
    parseErrorEndLine: int option
    parseErrorEndCol:  int option
    parseErrors:       SerializedParseError array
    assignments:       AssignmentHint array
}

// ---------------------------------------------------------------------------
// Workspace injection: the TS host provides file content, not System.IO.
// ---------------------------------------------------------------------------

/// Parse MATLAB source and try to load a function body for cross-file analysis.
let private tryParseExternalBody (source: string) : (FunctionSignature * Map<string, FunctionSignature>) option =
    try
        let (program, _) = Parser.parseMATLAB source
        let funcDefs =
            program.body
            |> List.choose (fun stmt ->
                match stmt with
                | FunctionDef({ line = line; col = col }, name, parms, outputVars, body, argAnns) ->
                    Some { name = name; parms = parms; outputVars = outputVars; body = body
                           defLine = line; defCol = col; argShapes = Shapes.argAnnotationsToShapes argAnns }
                | _ -> None)
        match funcDefs with
        | [] -> None
        | primary :: rest ->
            let subfunctions = rest |> List.map (fun s -> (s.name, s)) |> Map.ofList
            Some (primary, subfunctions)
    with _ -> None

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
            // Try to parse the full body for cross-file analysis
            let bodyOpt = tryParseExternalBody content
            let (body, parmNames, outputNames) =
                match bodyOpt with
                | Some (primary, _) -> (Some primary.body, primary.parms, primary.outputVars)
                | None -> (None, [], [])
            result.[key] <- {
                filename    = funcName
                paramCount  = paramCount
                returnCount = returnCount
                sourcePath  = ""
                body        = body
                parmNames   = parmNames
                outputNames = outputNames
            }
    result |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq

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
    let parseResult =
        try Some (Parser.parseMATLAB source)
        with
        | Parser.ParseError(_, _, _, _, _) -> None
        | Lexer.LexError(_, _, _, _, _) -> None
        | _ -> None

    match parseResult with
    | None ->
        // Return parse error — TS server will create a diagnostic for it
        let (errMsg, errLine, errCol, errEndLine, errEndCol) =
            try
                Parser.parseMATLAB source |> ignore
                ("Unknown parse error", None, None, None, None)
            with
            | Parser.ParseError(msg, line, col, endLine, endCol) -> (msg, Some line, Some col, Some endLine, Some endCol)
            | Lexer.LexError(msg, line, col, endLine, endCol) -> (msg, Some line, Some col, Some endLine, Some endCol)
            | ex -> (ex.Message, None, None, None, None)
        { diagnostics = [||]; env = [||]; symbols = [||]; parseError = Some errMsg
          parseErrorLine = errLine; parseErrorCol = errCol
          parseErrorEndLine = errEndLine; parseErrorEndCol = errEndCol
          parseErrors = [||]; assignments = [||] }

    | Some (irProg, recoveredParseErrors) ->

    // Serialize recovered parse errors
    let serializedParseErrors =
        recoveredParseErrors
        |> List.map (fun pe ->
            { message = pe.message; startLine = pe.startLine; startCol = pe.startCol
              endLine = pe.endLine; endCol = pe.endCol })
        |> Array.ofList

    // Extract symbols (function definitions) from IR
    let symbols =
        irProg.body
        |> List.choose (fun stmt ->
            match stmt with
            | FunctionDef({ line = line; col = col }, name, parms, outputVars, _, _) ->
                Some { name = name; line = line; col = col; parms = Array.ofList parms; outputs = Array.ofList outputVars }
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

    // Filter: apply suppression directives then strict-only filter
    let suppressions = Suppressions.parseSuppressions source
    let displayWarnings =
        warnings
        |> Suppressions.filterDiagnostics suppressions
        |> (if strict then id else List.filter (fun w -> not (Set.contains w.code STRICT_ONLY_CODES)))

    // Serialize diagnostics
    let diags =
        displayWarnings
        |> List.map (fun w ->
            { line = w.line; col = w.col; code = codeString w.code; message = w.message
              relatedLine = w.relatedLine; relatedCol = w.relatedCol })
        |> Array.ofList

    // Serialize environment
    let envPairs =
        env.bindings
        |> Map.toArray
        |> Array.map (fun (k, v) -> (k, shapeToString v))

    // Collect first-assignment hints (same algorithm as LspInlayHints.fs)
    let seen = System.Collections.Generic.HashSet<string>()
    let hintList = ResizeArray<AssignmentHint>()

    let tryEmitHint (loc: Ir.SrcLoc) (name: string) =
        if seen.Add(name) then
            let shape = Env.Env.get env name
            match shape with
            | Bottom | UnknownShape -> ()
            | _ -> hintList.Add({ name = name; line = loc.line; col = loc.col; shape = shapeToString shape })

    let rec walkStmts stmts =
        for stmt in stmts do walkStmt stmt
    and walkStmt stmt =
        match stmt with
        | Ir.Assign(loc, name, _) -> tryEmitHint loc name
        | Ir.AssignMulti(loc, targets, _) ->
            for name in targets do tryEmitHint loc name
        | Ir.For(loc, var_, _, body) -> tryEmitHint loc var_; walkStmts body
        | Ir.If(_, _, tb, eb) -> walkStmts tb; walkStmts eb
        | Ir.IfChain(_, _, bodies, eb) -> for b in bodies do walkStmts b; walkStmts eb
        | Ir.While(_, _, body) -> walkStmts body
        | Ir.Switch(_, _, cases, ow) -> for (_, cb) in cases do walkStmts cb; walkStmts ow
        | Ir.Try(_, tb, cb) -> walkStmts tb; walkStmts cb
        | Ir.FunctionDef _ -> ()
        | _ -> ()

    walkStmts irProg.body

    { diagnostics = diags; env = envPairs; symbols = symbols; parseError = None
      parseErrorLine = None; parseErrorCol = None
      parseErrorEndLine = None; parseErrorEndCol = None
      parseErrors = serializedParseErrors; assignments = hintList.ToArray() }
