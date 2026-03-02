module TestRunner

open System
open System.IO
open System.Text.RegularExpressions
open Context
open Diagnostics
open Analysis
open Workspace
open WarningCodes

// ---------------------------------------------------------------------------
// Expectation parsing
// ---------------------------------------------------------------------------

let private expectRe            = Regex(@"%\s*EXPECT:\s*(.+)$",                       RegexOptions.Multiline)
let private expectFixpointRe    = Regex(@"%\s*EXPECT_FIXPOINT:\s*(.+)$",              RegexOptions.Multiline)
let private expectWarningsRe    = Regex(@"warnings\s*(=|>=|>|<=|<)\s*(\d+)\s*$",      RegexOptions.IgnoreCase)
let private expectBindingRe     = Regex(@"([A-Za-z_]\w*)\s*=\s*(.+)$")
let private modeCoderRe         = Regex(@"%\s*MODE:\s*coder",                          RegexOptions.Multiline)
let private expectWarningRe     = Regex(@"%\s*EXPECT_WARNING:\s*(W_\w+)",              RegexOptions.Multiline)
let private expectNoWarningRe   = Regex(@"%\s*EXPECT_NO_WARNING:\s*(W_\w+)",           RegexOptions.Multiline)
let private expectFpWarningRe   = Regex(@"%\s*EXPECT_FIXPOINT_WARNING:\s*(W_\w+)",    RegexOptions.Multiline)
let private expectFpNoWarningRe = Regex(@"%\s*EXPECT_FIXPOINT_NO_WARNING:\s*(W_\w+)", RegexOptions.Multiline)

let private normalizeShapeStr (s: string) : string =
    Regex.Replace(s.Trim(), @"\s+", "")

/// Convert a character offset in src to a 1-based line number.
let private lineOfOffset (src: string) (offset: int) : int =
    let mutable count = 1
    for i in 0 .. (min offset (src.Length - 1)) - 1 do
        if src.[i] = '\n' then count <- count + 1
    count

type private WarningOp = Eq | Ge | Gt | Le | Lt

let private parseWarningOp (s: string) : WarningOp =
    match s with
    | ">=" -> Ge | ">" -> Gt | "<=" -> Le | "<" -> Lt | _ -> Eq

let private warningOpStr (op: WarningOp) : string =
    match op with
    | Eq -> "=" | Ge -> ">=" | Gt -> ">" | Le -> "<=" | Lt -> "<"

type private Expectations = {
    shapes:         Map<string, string>     // varname -> normalized shape string
    warningCheck:   (WarningOp * int) option
    expectWarnings: (int * string) list     // (line, code) pairs: warning MUST fire here
    rejectWarnings: (int * string) list     // (line, code) pairs: warning must NOT fire here
}

let private parseExpectations (src: string) (fixpoint: bool) : Expectations =
    let mutable shapes        = Map.empty<string, string>
    let mutable warningCheck  : (WarningOp * int) option = None
    let mutable fpShapes      = Map.empty<string, string>
    let mutable fpWarnCheck   : (WarningOp * int) option = None
    let mutable hasFp         = false

    // Parse EXPECT_FIXPOINT: lines
    for m in expectFixpointRe.Matches(src) do
        hasFp <- true
        let payload = m.Groups.[1].Value.Trim()
        let mw = expectWarningsRe.Match(payload)
        if mw.Success then
            fpWarnCheck <- Some (parseWarningOp mw.Groups.[1].Value, int mw.Groups.[2].Value)
        else
            let mb = expectBindingRe.Match(payload)
            if mb.Success then
                fpShapes <- Map.add mb.Groups.[1].Value (normalizeShapeStr mb.Groups.[2].Value) fpShapes

    // Parse EXPECT: lines
    for m in expectRe.Matches(src) do
        let payload = m.Groups.[1].Value.Trim()
        let mw = expectWarningsRe.Match(payload)
        if mw.Success then
            warningCheck <- Some (parseWarningOp mw.Groups.[1].Value, int mw.Groups.[2].Value)
        else
            let mb = expectBindingRe.Match(payload)
            if mb.Success then
                shapes <- Map.add mb.Groups.[1].Value (normalizeShapeStr mb.Groups.[2].Value) shapes

    // In fixpoint mode, override with fixpoint-specific expectations
    if fixpoint && hasFp then
        for kv in fpShapes do
            shapes <- Map.add kv.Key kv.Value shapes
        match fpWarnCheck with
        | Some c -> warningCheck <- Some c
        | None   -> ()

    // Parse inline EXPECT_WARNING / EXPECT_NO_WARNING directives
    let mutable expectWarns   : (int * string) list = []
    let mutable rejectWarns   : (int * string) list = []
    let mutable fpExpectWarns : (int * string) list = []
    let mutable fpRejectWarns : (int * string) list = []
    let mutable hasFpWarns    = false

    let parseInlineCode (label: string) (m: Text.RegularExpressions.Match) : (int * string) option =
        let code = m.Groups.[1].Value
        match tryParseCode code with
        | None ->
            printfn "PARSE ERROR: unknown warning code '%s' in %s" code label
            Some (-1, code)
        | Some _ ->
            Some (lineOfOffset src m.Index, code)

    for m in expectFpWarningRe.Matches(src) do
        hasFpWarns <- true
        match parseInlineCode "EXPECT_FIXPOINT_WARNING" m with
        | Some pair -> fpExpectWarns <- pair :: fpExpectWarns
        | None      -> ()

    for m in expectFpNoWarningRe.Matches(src) do
        hasFpWarns <- true
        match parseInlineCode "EXPECT_FIXPOINT_NO_WARNING" m with
        | Some pair -> fpRejectWarns <- pair :: fpRejectWarns
        | None      -> ()

    for m in expectWarningRe.Matches(src) do
        match parseInlineCode "EXPECT_WARNING" m with
        | Some pair -> expectWarns <- pair :: expectWarns
        | None      -> ()

    for m in expectNoWarningRe.Matches(src) do
        match parseInlineCode "EXPECT_NO_WARNING" m with
        | Some pair -> rejectWarns <- pair :: rejectWarns
        | None      -> ()

    // In fixpoint mode, FIXPOINT_WARNING variants completely replace non-fixpoint ones
    if fixpoint && hasFpWarns then
        expectWarns <- fpExpectWarns
        rejectWarns <- fpRejectWarns

    { shapes = shapes; warningCheck = warningCheck
      expectWarnings = expectWarns; rejectWarnings = rejectWarns }


// ---------------------------------------------------------------------------
// Single test runner
// ---------------------------------------------------------------------------

/// runTest: analyze one .m file, check assertions.
/// Returns (passed: bool).
let private runTest (path: string) (fixpoint: bool) (forceCoder: bool) : bool =
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

    // Determine coder mode: forced by directory or by % MODE: coder directive
    let coderMode = forceCoder || modeCoderRe.IsMatch(src)

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
    ctx.cst.coderMode <- coderMode
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
    match expectations.warningCheck with
    | Some (op, expected) ->
        let actual = warnings.Length
        let pass =
            match op with
            | Eq -> actual = expected
            | Ge -> actual >= expected
            | Gt -> actual > expected
            | Le -> actual <= expected
            | Lt -> actual < expected
        if not pass then
            printfn "ASSERT FAIL: expected warnings %s %d, got %d" (warningOpStr op) expected actual
            passed <- false
    | None -> ()

    // Check variable shapes
    for kv in expectations.shapes do
        let varName     = kv.Key
        let expectedStr = kv.Value
        let actualShape = Env.Env.get env varName
        let actualStr   = normalizeShapeStr (Shapes.shapeToString actualShape)
        if actualStr <> expectedStr then
            printfn "ASSERT FAIL: expected %s = %s, got %s" varName expectedStr actualStr
            passed <- false

    // Check line-specific warning code assertions
    let warningSet =
        warnings |> List.map (fun w -> (w.line, codeString w.code)) |> Set.ofList

    for (line, code) in expectations.expectWarnings do
        if line = -1 then
            // Parse error for unknown code -- already printed, just fail
            passed <- false
        elif not (Set.contains (line, code) warningSet) then
            printfn "ASSERT FAIL: expected %s on line %d, not found" code line
            passed <- false

    for (line, code) in expectations.rejectWarnings do
        if line = -1 then
            passed <- false
        elif Set.contains (line, code) warningSet then
            printfn "ASSERT FAIL: unexpected %s on line %d" code line
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
let run (strict: bool) (fixpoint: bool) (coder: bool) : int =
    // Discover tests relative to cwd (project root when running via `dotnet run`)
    let testsDir =
        let cwd = Directory.GetCurrentDirectory()
        let candidate = Path.Combine(cwd, "tests")
        if Directory.Exists candidate then candidate
        else
            // Try parent of src/ directory
            let parent = Path.GetDirectoryName(cwd)
            Path.Combine(parent, "tests")

    let coderDir = Path.Combine(testsDir, "coder")

    let testFiles = discoverTestFiles testsDir

    if testFiles.IsEmpty then
        eprintfn "WARNING: No test files found under %s" testsDir

    let mutable total = 0
    let mutable ok    = 0

    for path in testFiles do
        total <- total + 1
        // Auto-enable coder mode for tests/coder/ directory
        let inCoderDir = path.StartsWith(coderDir + Path.DirectorySeparatorChar.ToString()) || path = coderDir
        let passed = runTest path fixpoint (coder || inCoderDir)
        if passed then ok <- ok + 1

    printfn "===== Summary: %d/%d tests passed =====" ok total

    let rc = if ok = total then 0 else 1

    if strict then
        // In strict mode, exit with error if any unsupported constructs found
        // (the test runner itself doesn't re-check for unsupported, just reports summary)
        rc
    else
        rc
