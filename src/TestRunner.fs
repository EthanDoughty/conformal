module TestRunner

open System
open System.IO
open System.Text.RegularExpressions
open Context
open Diagnostics
open Analysis
open Workspace

// ---------------------------------------------------------------------------
// Expectation parsing
// ---------------------------------------------------------------------------

let private expectRe          = Regex(@"%\s*EXPECT:\s*(.+)$",          RegexOptions.Multiline)
let private expectFixpointRe  = Regex(@"%\s*EXPECT_FIXPOINT:\s*(.+)$", RegexOptions.Multiline)
let private expectWarningsRe  = Regex(@"warnings\s*=\s*(\d+)\s*$",     RegexOptions.IgnoreCase)
let private expectBindingRe   = Regex(@"([A-Za-z_]\w*)\s*=\s*(.+)$")

let private normalizeShapeStr (s: string) : string =
    Regex.Replace(s.Trim(), @"\s+", "")

type private Expectations = {
    shapes:       Map<string, string>  // varname -> normalized shape string
    warningCount: int option
}

let private parseExpectations (src: string) (fixpoint: bool) : Expectations =
    let mutable shapes        = Map.empty<string, string>
    let mutable warningCount  = None
    let mutable fpShapes      = Map.empty<string, string>
    let mutable fpWarnCount   = None
    let mutable hasFp         = false

    // Parse EXPECT_FIXPOINT: lines
    for m in expectFixpointRe.Matches(src) do
        hasFp <- true
        let payload = m.Groups.[1].Value.Trim()
        let mw = expectWarningsRe.Match(payload)
        if mw.Success then
            fpWarnCount <- Some (int mw.Groups.[1].Value)
        else
            let mb = expectBindingRe.Match(payload)
            if mb.Success then
                fpShapes <- Map.add mb.Groups.[1].Value (normalizeShapeStr mb.Groups.[2].Value) fpShapes

    // Parse EXPECT: lines
    for m in expectRe.Matches(src) do
        let payload = m.Groups.[1].Value.Trim()
        let mw = expectWarningsRe.Match(payload)
        if mw.Success then
            warningCount <- Some (int mw.Groups.[1].Value)
        else
            let mb = expectBindingRe.Match(payload)
            if mb.Success then
                shapes <- Map.add mb.Groups.[1].Value (normalizeShapeStr mb.Groups.[2].Value) shapes

    // In fixpoint mode, override with fixpoint-specific expectations
    if fixpoint && hasFp then
        for kv in fpShapes do
            shapes <- Map.add kv.Key kv.Value shapes
        match fpWarnCount with
        | Some c -> warningCount <- Some c
        | None   -> ()

    { shapes = shapes; warningCount = warningCount }


// ---------------------------------------------------------------------------
// Single test runner
// ---------------------------------------------------------------------------

/// runTest: analyze one .m file, check assertions.
/// Returns (passed: bool).
let private runTest (path: string) (fixpoint: bool) : bool =
    printfn "===== Analysis for %s" path

    if not (File.Exists path) then
        printfn "ERROR: file not found"
        printfn ""
        false
    else

    let src =
        try File.ReadAllText(path)
        with ex ->
            eprintfn "Error reading %s: %s" path ex.Message
            ""

    let expectations = parseExpectations src fixpoint

    let irProgOpt =
        try
            let prog = Parser.parseMATLAB src
            Some prog
        with
        | Parser.ParseError msg ->
            printfn "Error while parsing %s: ParseError: %s" path msg
            None
        | Lexer.LexError msg ->
            printfn "Error while parsing %s: LexError: %s" path msg
            None
        | ex ->
            printfn "Error while parsing %s: %s" path ex.Message
            None

    match irProgOpt with
    | None ->
        printfn ""
        false
    | Some irProg ->

    let dirPath = Path.GetDirectoryName(Path.GetFullPath(path))
    let fileName = Path.GetFileName(path)
    let extMap = scanWorkspace dirPath fileName

    let ctx = AnalysisContext()
    ctx.call.fixpoint <- fixpoint
    for kv in extMap do
        ctx.ws.externalFunctions.[kv.Key] <- kv.Value
    ctx.ws.workspaceDir <- dirPath

    let analyzeResult =
        try
            let (e, w) = analyzeProgramIr irProg ctx
            Some (e, w)
        with ex ->
            printfn "Error while analyzing %s: %s" path ex.Message
            None

    match analyzeResult with
    | None ->
        printfn "ASSERTIONS: FAIL"
        printfn ""
        false
    | Some (env, warnings) ->

    // NOTE: tests see UNFILTERED warnings (strict-only filtering is presentation concern)
    if warnings.IsEmpty then
        printfn "No dimension warnings."
    else
        printfn "Warnings:"
        for w in warnings do
            printfn "- %s" (diagnosticToString w)

    printfn "Final environment:"
    let envPairs =
        env.bindings
        |> Map.toList
        |> List.sortBy fst
        |> List.map (fun (k, v) -> k + ": " + Shapes.shapeToString v)
    printfn "    Env{%s}" (String.concat ", " envPairs)

    let mutable passed = true

    // Check warning count (unfiltered)
    match expectations.warningCount with
    | Some expected when expected <> warnings.Length ->
        printfn "ASSERT FAIL: expected warnings = %d, got %d" expected warnings.Length
        passed <- false
    | _ -> ()

    // Check variable shapes
    for kv in expectations.shapes do
        let varName     = kv.Key
        let expectedStr = kv.Value
        let actualShape = Env.Env.get env varName
        let actualStr   = normalizeShapeStr (Shapes.shapeToString actualShape)
        if actualStr <> expectedStr then
            printfn "ASSERT FAIL: expected %s = %s, got %s" varName expectedStr actualStr
            passed <- false

    printfn "ASSERTIONS: %s" (if passed then "PASS" else "FAIL")
    printfn ""
    passed


// ---------------------------------------------------------------------------
// Discover test files
// ---------------------------------------------------------------------------

/// Discover all .m test files under the given root directory, sorted.
let private discoverTestFiles (rootDir: string) : string list =
    if not (Directory.Exists rootDir) then []
    else
        Directory.GetFiles(rootDir, "*.m", SearchOption.AllDirectories)
        |> Array.sort
        |> Array.toList


// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// run: run all tests, return exit code (0 = all pass, 1 = any fail).
let run (strict: bool) (fixpoint: bool) : int =
    // Discover tests relative to cwd (project root when running via `dotnet run`)
    let testsDir =
        let cwd = Directory.GetCurrentDirectory()
        let candidate = Path.Combine(cwd, "tests")
        if Directory.Exists candidate then candidate
        else
            // Try parent of src/ directory
            let parent = Path.GetDirectoryName(cwd)
            Path.Combine(parent, "tests")

    let testFiles = discoverTestFiles testsDir

    if testFiles.IsEmpty then
        eprintfn "WARNING: No test files found under %s" testsDir

    let mutable total = 0
    let mutable ok    = 0

    for path in testFiles do
        total <- total + 1
        let passed = runTest path fixpoint
        if passed then ok <- ok + 1

    printfn "===== Summary: %d/%d tests passed =====" ok total

    let rc = if ok = total then 0 else 1

    if strict then
        // In strict mode, exit with error if any unsupported constructs found
        // (the test runner itself doesn't re-check for unsupported, just reports summary)
        rc
    else
        rc
