module Cli

open System
open System.IO
open Context
open Diagnostics
open WarningCodes
open Analysis
open Workspace

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

// Format a single diagnostic for CLI output.
// Python format: "W_CODE line N: message" or "W_UNSUPPORTED_STMT line=N targets=..."
let private formatDiag (d: Diagnostic) : string =
    diagnosticToString d

// Print the final environment in Env{...} format (sorted by key).
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

/// Analyze one .m file and print results. Returns exit code: 0 = success, 1 = error.
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
                let (prog, _parseErrs) = Parser.parseMATLAB src
                Some prog
            with
            | Parser.ParseError(msg, _, _, _, _) ->
                eprintfn "Error while parsing %s: ParseError: %s" filePath msg
                None
            | Lexer.LexError(msg, _, _, _, _) ->
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
        let (flatFuncs, flatClasses) = scanWorkspace dirPath fileName 0
        let addpathDirs = extractAddpathDirs irProg
        let (addpathFuncs, addpathClasses) = scanAddpathDirs dirPath addpathDirs fileName
        let (depthFuncs, depthClasses) = scanWorkspace dirPath fileName 3
        let extMap = mergeMaps [flatFuncs; addpathFuncs; depthFuncs]
        let classdefMap = mergeMaps [flatClasses; addpathClasses; depthClasses]
        let privateMap = scanPrivateDir dirPath

        let tScan = DateTime.UtcNow

        let ctx = AnalysisContext()
        ctx.call.fixpoint <- fixpoint
        ctx.cst.coderMode <- coder
        for kv in extMap do
            ctx.ws.externalFunctions.[kv.Key] <- kv.Value
        for kv in classdefMap do
            ctx.ws.externalClassdefs.[kv.Key] <- kv.Value
        for kv in privateMap do
            ctx.ws.privateFunctions.[kv.Key] <- kv.Value
        ctx.ws.workspaceDir <- dirPath

        let (env, warnings) = analyzeProgramIr irProg ctx

        let tAnalyze = DateTime.UtcNow

        // Filter: apply suppression directives then strict-only filter
        let suppressions = Suppressions.parseSuppressions src
        let displayWarnings =
            warnings
            |> Suppressions.filterDiagnostics suppressions
            |> (if strict then id else List.filter (fun w -> not (Set.contains w.code STRICT_ONLY_CODES)))

        printfn "=== Analysis for %s ===" filePath
        if displayWarnings.IsEmpty then
            printfn "No dimension warnings."
        else
            printfn "Warnings:"
            for w in displayWarnings do
                printfn "  - %s" (formatDiag w)

        printfn ""
        printEnv env

        let (tracked, partial, untracked, total) = computeShapeCoverage env
        if untracked > 0 || partial > 0 then
            let unresolvedFns =
                warnings
                |> List.filter (fun w -> w.code = W_UNKNOWN_FUNCTION)
                |> List.length
            printfn ""
            printfn "Shape coverage: %d/%d tracked, %d partial, %d untracked (%d unresolved functions)"
                tracked total partial untracked unresolvedFns

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

let runTests (strict: bool) (fixpoint: bool) (benchmark: bool) (quiet: bool) : int =
    let tStart = DateTime.UtcNow
    let result = TestRunner.run strict fixpoint false quiet
    let tEnd = DateTime.UtcNow

    if benchmark then
        let totalMs = (tEnd - tStart).TotalMilliseconds
        printfn ""
        printfn "--- Benchmark ---"
        printfn "  Total:     %.0fms" totalMs
        printfn "  Mode:      %s" (if fixpoint then "fixpoint" else "normal")

    result

// ---------------------------------------------------------------------------
// Argument parsing and entry point
// ---------------------------------------------------------------------------

type CliArgs = {
    tests: bool; testProps: bool; strict: bool; fixpoint: bool
    bench: bool; coder: bool; file: string; parseJson: bool
    quiet: bool; help: bool; version: bool; formatSarif: bool
}

let private defaultArgs =
    { tests = false; testProps = false; strict = false; fixpoint = false
      bench = false; coder = false; file = ""; parseJson = false
      quiet = false; help = false; version = false; formatSarif = false }

// Fold state: Ready accepts flags, ConsumeFile means next arg is a file path,
// ConsumeWitness means next arg is an optional witness mode or file path.
type private ParseState = Ready | ConsumeFile | ConsumeWitness | ConsumeFormat

let private parseArgv (argv: string array) : CliArgs =
    argv
    |> Array.fold (fun (acc, state) arg ->
        match state with
        | ConsumeFile ->
            ({ acc with file = arg }, Ready)
        | ConsumeFormat ->
            if arg = "sarif" then
                ({ acc with formatSarif = true }, Ready)
            else
                (acc, Ready) // unknown format, ignore
        | ConsumeWitness ->
            if arg.StartsWith("--") then
                // Not a mode/file arg; re-parse as a flag on the next fold iteration won't work
                // since fold has already consumed it. Handle known flags inline.
                match arg with
                | "--tests"        -> ({ acc with tests = true }, Ready)
                | "--test-props"   -> ({ acc with testProps = true }, Ready)
                | "--strict"       -> ({ acc with strict = true }, Ready)
                | "--fixpoint"     -> ({ acc with fixpoint = true }, Ready)
                | "--benchmark"    -> ({ acc with bench = true }, Ready)
                | "--coder"        -> ({ acc with coder = true }, Ready)
                | "--parse-json"   -> ({ acc with parseJson = true }, ConsumeFile)
                | "--format"       -> (acc, ConsumeFormat)
                | "--witness"      -> (acc, ConsumeWitness)
                | "--quiet"        -> ({ acc with quiet = true }, Ready)
                | "--help" | "-h"  -> ({ acc with help = true }, Ready)
                | "--version"      -> ({ acc with version = true }, Ready)
                | _ -> (acc, Ready)
            else
                match arg with
                | "enrich" | "filter" | "tag" -> (acc, Ready)  // witness mode string, ignore
                | f -> ({ acc with file = f }, Ready)
        | Ready ->
            match arg with
            | "--tests"        -> ({ acc with tests = true }, Ready)
            | "--test-props"   -> ({ acc with testProps = true }, Ready)
            | "--strict"       -> ({ acc with strict = true }, Ready)
            | "--fixpoint"     -> ({ acc with fixpoint = true }, Ready)
            | "--benchmark"    -> ({ acc with bench = true }, Ready)
            | "--coder"        -> ({ acc with coder = true }, Ready)
            | "--parse-json"   -> ({ acc with parseJson = true }, ConsumeFile)
            | "--format"       -> (acc, ConsumeFormat)
            | "--witness"      -> (acc, ConsumeWitness)
            | "--quiet"        -> ({ acc with quiet = true }, Ready)
            | "--help" | "-h"  -> ({ acc with help = true }, Ready)
            | "--version"      -> ({ acc with version = true }, Ready)
            | a when not (a.StartsWith("--")) -> ({ acc with file = a }, Ready)
            | _ -> (acc, Ready)
    ) (defaultArgs, Ready) |> fst

let private printUsage () =
    printfn "Usage: conformal [OPTIONS] <file.m>"
    printfn ""
    printfn "Options:"
    printfn "  --tests         Run test suite"
    printfn "  --test-props    Run property-based tests (FsCheck)"
    printfn "  --strict        Show all warnings including informational diagnostics"
    printfn "  --fixpoint      Use fixed-point iteration for loop analysis"
    printfn "  --benchmark     Print timing breakdown"
    printfn "  --coder         Enable MATLAB Coder compatibility warnings (W_CODER_*)"
    printfn "  --parse-json    Parse file and emit JSON IR"
    printfn "  --format sarif  Output diagnostics as SARIF 2.1.0 JSON"
    printfn "  --help, -h      Show this help message"
    printfn "  --version       Show version"

/// Analyze one .m file and emit SARIF 2.1.0 JSON to stdout. Returns exit code.
let runFileSarif (filePath: string) (strict: bool) (fixpoint: bool) (coder: bool) : int =
    if not (File.Exists filePath) then
        eprintfn "ERROR: file not found: %s" filePath
        1
    else
        let src =
            try File.ReadAllText(filePath)
            with ex ->
                eprintfn "Error reading %s: %s" filePath ex.Message
                ""

        if src = "" && not (File.Exists filePath) then 1
        else

        let irProgOpt =
            try
                let (prog, _parseErrs) = Parser.parseMATLAB src
                Some prog
            with
            | Parser.ParseError(msg, _, _, _, _) ->
                eprintfn "Error while parsing %s: ParseError: %s" filePath msg
                None
            | Lexer.LexError(msg, _, _, _, _) ->
                eprintfn "Error while parsing %s: LexError: %s" filePath msg
                None
            | ex ->
                eprintfn "Error while parsing %s: %s" filePath ex.Message
                None

        match irProgOpt with
        | None -> 1
        | Some irProg ->

        let dirPath = Path.GetDirectoryName(Path.GetFullPath(filePath))
        let fileName = Path.GetFileName(filePath)
        let (flatFuncs, flatClasses) = Workspace.scanWorkspace dirPath fileName 0
        let addpathDirs = Workspace.extractAddpathDirs irProg
        let (addpathFuncs, addpathClasses) = Workspace.scanAddpathDirs dirPath addpathDirs fileName
        let (depthFuncs, depthClasses) = Workspace.scanWorkspace dirPath fileName 3
        let extMap = Workspace.mergeMaps [flatFuncs; addpathFuncs; depthFuncs]
        let classdefMap = Workspace.mergeMaps [flatClasses; addpathClasses; depthClasses]
        let privateMap = Workspace.scanPrivateDir dirPath

        let ctx = Context.AnalysisContext()
        ctx.call.fixpoint <- fixpoint
        ctx.cst.coderMode <- coder
        for kv in extMap do
            ctx.ws.externalFunctions.[kv.Key] <- kv.Value
        for kv in classdefMap do
            ctx.ws.externalClassdefs.[kv.Key] <- kv.Value
        for kv in privateMap do
            ctx.ws.privateFunctions.[kv.Key] <- kv.Value
        ctx.ws.workspaceDir <- dirPath

        let (_env, warnings) = Analysis.analyzeProgramIr irProg ctx

        // Filter: apply suppression directives then strict-only filter
        let suppressions = Suppressions.parseSuppressions src
        let displayWarnings =
            warnings
            |> Suppressions.filterDiagnostics suppressions
            |> (if strict then id else List.filter (fun w -> not (Set.contains w.code Diagnostics.STRICT_ONLY_CODES)))

        // Compute relative URI from CWD
        let cwd = Directory.GetCurrentDirectory()
        let fullPath = Path.GetFullPath(filePath)
        let relUri =
            if fullPath.StartsWith(cwd + string Path.DirectorySeparatorChar) then
                fullPath.Substring(cwd.Length + 1)
            elif fullPath.StartsWith(cwd) && cwd.Length = fullPath.Length then
                fileName
            else
                filePath
        let relUri = relUri.Replace('\\', '/')

        use stream = Console.OpenStandardOutput()
        SarifEmitter.emitSarif stream relUri displayWarnings "3.5.0"
        // Write trailing newline so shell prompt starts on new line
        stream.WriteByte(10uy)
        0

/// Parse argv and dispatch. Returns exit code.
let run (argv: string array) : int =
    let args = parseArgv argv

    if args.version then
        printfn "Conformal 3.5.0"
        0
    elif args.help then
        printUsage ()
        0
    elif args.testProps then
        PropertyTests.runPropertyTests()
    elif args.tests then
        runTests args.strict args.fixpoint args.bench args.quiet
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
                let (program, _) = Parser.parseMATLAB src
                let json = Json.programToJson program
                printfn "%s" json
                0
            with
            | Parser.ParseError(msg, _, _, _, _) ->
                eprintfn "ParseError: %s" msg
                2
            | Lexer.LexError(msg, _, _, _, _) ->
                eprintfn "LexError: %s" msg
                2
            | ex ->
                eprintfn "Error: %s" ex.Message
                3
    elif args.formatSarif && args.file <> "" then
        runFileSarif args.file args.strict args.fixpoint args.coder
    elif args.file <> "" then
        runFile args.file args.strict args.fixpoint args.bench args.coder
    else
        printUsage ()
        1
