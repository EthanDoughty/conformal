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
        |> List.map (fun (k, v) -> $"{k}: {Shapes.shapeToString v}")
    let inner = String.concat ", " pairs
    let envStr = $"Env{{{inner}}}"
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

// ---------------------------------------------------------------------------
// Argument parsing types
// ---------------------------------------------------------------------------

type CliArgs = {
    tests: bool; testProps: bool; strict: bool; fixpoint: bool
    bench: bool; coder: bool; file: string; parseJson: bool
}

let private defaultArgs =
    { tests = false; testProps = false; strict = false; fixpoint = false
      bench = false; coder = false; file = ""; parseJson = false }

/// Fold state: Ready accepts flags, ConsumeFile means next arg is a file path,
/// ConsumeWitness means next arg is an optional witness mode or file path.
type private ParseState = Ready | ConsumeFile | ConsumeWitness

let private parseArgv (argv: string array) : CliArgs =
    argv
    |> Array.fold (fun (acc, state) arg ->
        match state with
        | ConsumeFile ->
            ({ acc with file = arg }, Ready)
        | ConsumeWitness ->
            if arg.StartsWith("--") then
                // Not a mode/file arg; re-parse as a flag on the next fold iteration won't work
                // since fold has already consumed it. Handle known flags inline.
                match arg with
                | "--tests"      -> ({ acc with tests = true }, Ready)
                | "--test-props" -> ({ acc with testProps = true }, Ready)
                | "--strict"     -> ({ acc with strict = true }, Ready)
                | "--fixpoint"   -> ({ acc with fixpoint = true }, Ready)
                | "--benchmark"  -> ({ acc with bench = true }, Ready)
                | "--coder"      -> ({ acc with coder = true }, Ready)
                | "--parse-json" -> ({ acc with parseJson = true }, ConsumeFile)
                | "--witness"    -> (acc, ConsumeWitness)
                | _ -> (acc, Ready)
            else
                match arg with
                | "enrich" | "filter" | "tag" -> (acc, Ready)  // witness mode string, ignore
                | f -> ({ acc with file = f }, Ready)
        | Ready ->
            match arg with
            | "--tests"      -> ({ acc with tests = true }, Ready)
            | "--test-props" -> ({ acc with testProps = true }, Ready)
            | "--strict"     -> ({ acc with strict = true }, Ready)
            | "--fixpoint"   -> ({ acc with fixpoint = true }, Ready)
            | "--benchmark"  -> ({ acc with bench = true }, Ready)
            | "--coder"      -> ({ acc with coder = true }, Ready)
            | "--parse-json" -> ({ acc with parseJson = true }, ConsumeFile)
            | "--witness"    -> (acc, ConsumeWitness)
            | a when not (a.StartsWith("--")) -> ({ acc with file = a }, Ready)
            | _ -> (acc, Ready)
    ) (defaultArgs, Ready) |> fst

/// run: parse argv and dispatch.
/// Returns exit code.
let run (argv: string array) : int =
    let args = parseArgv argv

    if args.testProps then
        PropertyTests.runPropertyTests()
    elif args.tests then
        runTests args.strict args.fixpoint args.bench
    elif args.parseJson then
        // Legacy --parse-json mode: just parse and emit JSON
        if args.file = "" then
            eprintfn "Usage: conformal-parse --parse-json <file.m>"
            1
        elif not (File.Exists args.file) then
            eprintfn "File not found: %s" args.file
            1
        else
            try
                let src = File.ReadAllText(args.file)
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
    elif args.file <> "" then
        runFile args.file args.strict args.fixpoint args.bench args.coder
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
