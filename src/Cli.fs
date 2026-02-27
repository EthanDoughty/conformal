module Cli

open System
open System.IO
open Context
open Diagnostics
open Analysis
open Workspace

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Format a single diagnostic for CLI output (mirrors Python Diagnostic __str__).
/// Python format: "W_CODE line N: message" or "W_UNSUPPORTED_STMT line=N targets=..."
let private formatDiag (d: Diagnostic) : string =
    diagnosticToString d

/// Print the final environment in Env{...} format (sorted by key).
/// Mirrors Python Env.__repr__ but sorted alphabetically.
let private printEnv (env: Env.Env) : unit =
    let pairs =
        env.bindings
        |> Map.toList
        |> List.sortBy fst
        |> List.map (fun (k, v) -> k + ": " + Shapes.shapeToString v)
    let envStr = "Env{" + String.concat ", " pairs + "}"
    printfn "Final environment:"
    printfn "%s" envStr

// ---------------------------------------------------------------------------
// Single-file analysis
// ---------------------------------------------------------------------------

/// runFile: analyze one .m file and print results.
/// Returns exit code: 0 = success, 1 = error.
let runFile (filePath: string) (strict: bool) (fixpoint: bool) (benchmark: bool) (coder: bool) : int =
    if not (File.Exists filePath) then
        eprintfn "ERROR: file not found: %s" filePath
        1
    else
        let tStart = DateTime.UtcNow

        let src =
            try File.ReadAllText(filePath)
            with ex ->
                eprintfn "Error reading %s: %s" filePath ex.Message
                ""

        if src = "" && not (File.Exists filePath) then 1
        else

        let tRead = DateTime.UtcNow

        let irProgOpt =
            try
                let prog = Parser.parseMATLAB src
                Some prog
            with
            | Parser.ParseError msg ->
                eprintfn "Error while parsing %s: ParseError: %s" filePath msg
                None
            | Lexer.LexError msg ->
                eprintfn "Error while parsing %s: LexError: %s" filePath msg
                None
            | ex ->
                eprintfn "Error while parsing %s: %s" filePath ex.Message
                None

        match irProgOpt with
        | None -> 1
        | Some irProg ->

        let tParse = DateTime.UtcNow

        let dirPath = Path.GetDirectoryName(Path.GetFullPath(filePath))
        let fileName = Path.GetFileName(filePath)
        let extMap = scanWorkspace dirPath fileName

        let tScan = DateTime.UtcNow

        let ctx = AnalysisContext()
        ctx.call.fixpoint <- fixpoint
        ctx.cst.coderMode <- coder
        for kv in extMap do
            ctx.ws.externalFunctions.[kv.Key] <- kv.Value
        ctx.ws.workspaceDir <- dirPath

        let (env, warnings) = analyzeProgramIr irProg ctx

        let tAnalyze = DateTime.UtcNow

        // Filter strict-only codes in default mode
        let displayWarnings =
            if strict then warnings
            else warnings |> List.filter (fun w -> not (Set.contains w.code STRICT_ONLY_CODES))

        printfn "=== Analysis for %s ===" filePath
        if displayWarnings.IsEmpty then
            printfn "No dimension warnings."
        else
            printfn "Warnings:"
            for w in displayWarnings do
                printfn "  - %s" (formatDiag w)

        printfn ""
        printEnv env

        if benchmark then
            let lineCount = src.Split('\n').Length
            let totalMs = (tAnalyze - tStart).TotalMilliseconds
            printfn ""
            printfn "--- Benchmark (%d lines, %d warnings) ---" lineCount displayWarnings.Length
            printfn "  Read:      %7.1fms" (tRead - tStart).TotalMilliseconds
            printfn "  Parse:     %7.1fms" (tParse - tRead).TotalMilliseconds
            printfn "  Workspace: %7.1fms" (tScan - tParse).TotalMilliseconds
            printfn "  Analyze:   %7.1fms" (tAnalyze - tScan).TotalMilliseconds
            printfn "  Total:     %7.1fms" totalMs
            if totalMs > 0.0 then
                printfn "  Throughput: %.0f lines/sec" (float lineCount / totalMs * 1000.0)

        if strict && hasUnsupported displayWarnings then
            printfn ""
            printfn "STRICT MODE: Unsupported constructs detected (W_UNSUPPORTED_*)"
            1
        else
            0

// ---------------------------------------------------------------------------
// --tests dispatch
// ---------------------------------------------------------------------------

let runTests (strict: bool) (fixpoint: bool) (benchmark: bool) : int =
    let tStart = DateTime.UtcNow
    let result = TestRunner.run strict fixpoint false
    let tEnd = DateTime.UtcNow

    if benchmark then
        let totalMs = (tEnd - tStart).TotalMilliseconds
        printfn ""
        printfn "--- Benchmark ---"
        printfn "  Total:     %.0fms" totalMs
        printfn "  Mode:      %s" (if fixpoint then "fixpoint" else "normal")

    result

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// run: parse argv and dispatch.
/// Returns exit code.
let run (argv: string array) : int =
    // Manual argv parsing (no external deps)
    let mutable tests     = false
    let mutable testProps = false
    let mutable strict    = false
    let mutable fixpoint  = false
    let mutable bench     = false
    let mutable coder     = false
    let mutable file      = ""
    let mutable parseJson = false

    let mutable i = 0
    while i < argv.Length do
        match argv.[i] with
        | "--tests"       -> tests     <- true
        | "--test-props"  -> testProps <- true
        | "--strict"      -> strict    <- true
        | "--fixpoint"    -> fixpoint <- true
        | "--benchmark"   -> bench    <- true
        | "--coder"       -> coder    <- true
        | "--parse-json"  ->
            parseJson <- true
            i <- i + 1
            if i < argv.Length then file <- argv.[i]
        | "--witness"     ->
            // consume optional mode argument
            i <- i + 1
            if i < argv.Length && not (argv.[i].StartsWith("--")) then
                match argv.[i] with
                | "enrich" | "filter" | "tag" -> ()  // mode string, ignore for F# port
                | f -> file <- f  // it's a file path
            else
                i <- i - 1
        | arg when not (arg.StartsWith("--")) ->
            file <- arg
        | _ -> ()
        i <- i + 1

    if testProps then
        PropertyTests.runPropertyTests()
    elif tests then
        runTests strict fixpoint bench
    elif parseJson then
        // Legacy --parse-json mode: just parse and emit JSON
        if file = "" then
            eprintfn "Usage: conformal-parse --parse-json <file.m>"
            1
        elif not (File.Exists file) then
            eprintfn "File not found: %s" file
            1
        else
            try
                let src = File.ReadAllText(file)
                let program = Parser.parseMATLAB src
                let json = Json.programToJson program
                printfn "%s" json
                0
            with
            | Parser.ParseError msg ->
                eprintfn "ParseError: %s" msg
                2
            | Lexer.LexError msg ->
                eprintfn "LexError: %s" msg
                2
            | ex ->
                eprintfn "Error: %s" ex.Message
                3
    elif file <> "" then
        runFile file strict fixpoint bench coder
    else
        printfn "Usage: conformal-parse [--tests] [--strict] [--fixpoint] [--benchmark] [--coder] <file.m>"
        printfn ""
        printfn "Options:"
        printfn "  --tests       Run test suite"
        printfn "  --test-props  Run property-based tests (FsCheck)"
        printfn "  --strict      Show all warnings including informational diagnostics"
        printfn "  --fixpoint    Use fixed-point iteration for loop analysis"
        printfn "  --benchmark   Print timing breakdown"
        printfn "  --coder       Enable MATLAB Coder compatibility warnings (W_CODER_*)"
        printfn "  --parse-json  Parse file and emit JSON IR"
        1
