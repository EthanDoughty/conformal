module Program

open System
open System.IO
open Shapes
open SymDim
open Diagnostics
open Builtins
open Context

// ---------------------------------------------------------------------------
// Smoke test for Phase 1: SymDim + Shapes + Env
// ---------------------------------------------------------------------------

let runShapesTest () : int =
    let mutable failures = 0

    let check (label: string) (expected: string) (actual: string) =
        if expected = actual then
            Console.WriteLine("  PASS  " + label + " => " + actual)
        else
            Console.Error.WriteLine("  FAIL  " + label + ": expected \"" + expected + "\" got \"" + actual + "\"")
            failures <- failures + 1

    Console.WriteLine("=== Shape string tests ===")
    check "Scalar"          "scalar"          (shapeToString Scalar)
    check "Matrix 3x4"      "matrix[3 x 4]"   (shapeToString (Matrix(Concrete 3, Concrete 4)))
    check "Matrix n x None" "matrix[n x None]" (shapeToString (Matrix(Symbolic (SymDim.var "n"), Unknown)))
    check "Matrix None x 1" "matrix[None x 1]" (shapeToString (Matrix(Unknown, Concrete 1)))
    check "String"          "string"           (shapeToString StringShape)
    check "FunctionHandle"  "function_handle"  (shapeToString (FunctionHandle None))
    check "UnknownShape"    "unknown"          (shapeToString UnknownShape)
    check "Bottom"          "bottom"           (shapeToString Bottom)
    check "Struct empty"    "struct{}"         (shapeToString (Struct([], false)))
    check "Cell 2x1"        "cell[2 x 1]"      (shapeToString (Cell(Concrete 2, Concrete 1, None)))
    check "Matrix n x 1"    "matrix[n x 1]"    (shapeToString (Matrix(Symbolic (SymDim.var "n"), Concrete 1)))

    Console.WriteLine("=== SymDim arithmetic tests ===")
    let n = SymDim.var "n"
    let m = SymDim.var "m"
    let three = SymDim.const' 3

    check "var n"     "n"      (SymDim.toString n)
    check "const 3"   "3"      (SymDim.toString three)
    check "zero"      "0"      (SymDim.toString (SymDim.zero ()))
    check "n+3"       "n+3"    (SymDim.toString (SymDim.add n three))
    check "b-a"       "-a+b"   (SymDim.toString (SymDim.sub (SymDim.var "b") (SymDim.var "a")))
    check "n/2"       "n/2"    (SymDim.toString (SymDim.div n 2))
    check "n+m"       "m+n"    (SymDim.toString (SymDim.add n m))

    Console.WriteLine("=== Dim join/widen tests ===")
    let dimToStr d = match d with Concrete n -> "Concrete " + string n | Symbolic _ -> "Symbolic" | Unknown -> "Unknown"
    check "joinDim 3 3"     "Concrete 3" (dimToStr (joinDim (Concrete 3) (Concrete 3)))
    check "joinDim 3 4"     "Unknown"    (dimToStr (joinDim (Concrete 3) (Concrete 4)))
    check "widenDim 3 3"    "Concrete 3" (dimToStr (widenDim (Concrete 3) (Concrete 3)))
    check "widenDim 3 4"    "Unknown"    (dimToStr (widenDim (Concrete 3) (Concrete 4)))
    check "addDim 3 4"      "Concrete 7" (dimToStr (addDim (Concrete 3) (Concrete 4)))

    Console.WriteLine("=== Env tests ===")
    let env = Env.Env.create ()
    Env.Env.set env "x" Scalar
    Env.Env.set env "A" (Matrix(Concrete 3, Concrete 4))
    check "env.get x"      "scalar"         (shapeToString (Env.Env.get env "x"))
    check "env.get A"      "matrix[3 x 4]"  (shapeToString (Env.Env.get env "A"))
    check "env.get z"      "bottom"          (shapeToString (Env.Env.get env "z"))
    check "hasLocal x"     "True"            (string (Env.Env.hasLocal env "x"))
    check "contains x"     "True"            (string (Env.Env.contains env "x"))
    check "contains z"     "False"           (string (Env.Env.contains env "z"))

    let child = Env.Env.pushScope env
    Env.Env.set child "y" StringShape
    check "child.get y"    "string"          (shapeToString (Env.Env.get child "y"))
    check "child.get x"    "scalar"          (shapeToString (Env.Env.get child "x"))
    check "child.hasLocal x" "False"         (string (Env.Env.hasLocal child "x"))

    Console.WriteLine("=== dimsDefinitelyConflict tests ===")
    check "conflict 3 4"    "True"  (string (dimsDefinitelyConflict (Concrete 3) (Concrete 4)))
    check "no conflict 3 3" "False" (string (dimsDefinitelyConflict (Concrete 3) (Concrete 3)))
    check "no conflict unk" "False" (string (dimsDefinitelyConflict Unknown (Concrete 3)))

    Console.WriteLine("")
    if failures = 0 then
        Console.WriteLine("All shape smoke tests PASSED")
        0
    else
        Console.Error.WriteLine(string failures + " shape smoke test(s) FAILED")
        1

// ---------------------------------------------------------------------------
// Smoke test for Phase 2: Diagnostics + Builtins + Context + PathConstraints
// ---------------------------------------------------------------------------

let runPhase2Test () : int =
    let mutable failures = 0

    let check (label: string) (expected: string) (actual: string) =
        if expected = actual then
            Console.WriteLine("  PASS  " + label + " => " + actual)
        else
            Console.Error.WriteLine("  FAIL  " + label + ": expected \"" + expected + "\" got \"" + actual + "\"")
            failures <- failures + 1

    let checkBool (label: string) (expected: bool) (actual: bool) =
        check label (string expected) (string actual)

    Console.WriteLine("=== Phase 2: STRICT_ONLY_CODES ===")
    check "STRICT_ONLY_CODES count" "19" (string STRICT_ONLY_CODES.Count)
    checkBool "W_UNSUPPORTED_STMT in strict"     true  (Set.contains "W_UNSUPPORTED_STMT" STRICT_ONLY_CODES)
    checkBool "W_UNKNOWN_FUNCTION in strict"     true  (Set.contains "W_UNKNOWN_FUNCTION" STRICT_ONLY_CODES)
    checkBool "W_TOO_MANY_INDICES in strict"     true  (Set.contains "W_TOO_MANY_INDICES" STRICT_ONLY_CODES)
    checkBool "W_INNER_DIM_MISMATCH not strict"  false (Set.contains "W_INNER_DIM_MISMATCH" STRICT_ONLY_CODES)
    checkBool "W_ELEMENTWISE_MISMATCH not strict" false (Set.contains "W_ELEMENTWISE_MISMATCH" STRICT_ONLY_CODES)

    Console.WriteLine("=== Phase 2: KNOWN_BUILTINS ===")
    // Python KNOWN_BUILTINS has 323 entries (count from source)
    let expectedCount = 323
    let actualCount   = KNOWN_BUILTINS.Count
    check "KNOWN_BUILTINS count" (string expectedCount) (string actualCount)
    checkBool "zeros in builtins"  true (Set.contains "zeros"  KNOWN_BUILTINS)
    checkBool "ones in builtins"   true (Set.contains "ones"   KNOWN_BUILTINS)
    checkBool "eye in builtins"    true (Set.contains "eye"    KNOWN_BUILTINS)
    checkBool "rand in builtins"   true (Set.contains "rand"   KNOWN_BUILTINS)
    checkBool "size in builtins"   true (Set.contains "size"   KNOWN_BUILTINS)
    checkBool "length in builtins" true (Set.contains "length" KNOWN_BUILTINS)
    checkBool "kron in builtins"   true (Set.contains "kron"   KNOWN_BUILTINS)
    checkBool "blkdiag in builtins" true (Set.contains "blkdiag" KNOWN_BUILTINS)
    checkBool "fclose in builtins" true (Set.contains "fclose" KNOWN_BUILTINS)

    Console.WriteLine("=== Phase 2: Diagnostic record + warn_* builders ===")
    // warnInnerDimMismatch via warnMatmulMismatch
    let leftE  = Ir.Var(1, 0, "A")
    let rightE = Ir.Var(1, 5, "B")
    let leftS  = Matrix(Concrete 3, Concrete 4)
    let rightS = Matrix(Concrete 5, Concrete 2)
    let d = warnMatmulMismatch 1 leftE rightE leftS rightS false
    check "W_INNER_DIM_MISMATCH code" "W_INNER_DIM_MISMATCH" d.code
    check "W_INNER_DIM_MISMATCH line" "1" (string d.line)
    checkBool "W_INNER_DIM_MISMATCH relatedLine None" true d.relatedLine.IsNone

    let dUnk = warnUnknownFunction 5 "myFunc"
    check "W_UNKNOWN_FUNCTION code"    "W_UNKNOWN_FUNCTION" dUnk.code
    check "W_UNKNOWN_FUNCTION line"    "5" (string dUnk.line)
    check "W_UNKNOWN_FUNCTION message" "Function 'myFunc' is not recognized; treating result as unknown" dUnk.message

    let dUnsup = warnUnsupportedStmt 3 "parfor" ["x"; "y"]
    check "W_UNSUPPORTED_STMT code"    "W_UNSUPPORTED_STMT" dUnsup.code
    check "W_UNSUPPORTED_STMT toString" "W_UNSUPPORTED_STMT line=3 targets=x, y 'parfor'" (diagnosticToString dUnsup)

    let dElem = warnElementwiseMismatch 7 ".+" leftE rightE leftS rightS
    check "W_ELEMENTWISE_MISMATCH code" "W_ELEMENTWISE_MISMATCH" dElem.code
    check "W_ELEMENTWISE_MISMATCH toString"
        "W_ELEMENTWISE_MISMATCH line 7: Elementwise .+ mismatch in (A .+ B): matrix[3 x 4] vs matrix[5 x 2]"
        (diagnosticToString dElem)

    Console.WriteLine("=== Phase 2: hasUnsupported ===")
    let diags = [ dUnsup; dUnk ]
    checkBool "hasUnsupported true"  true  (hasUnsupported diags)
    checkBool "hasUnsupported false" false (hasUnsupported [ dUnk ])

    Console.WriteLine("=== Phase 2: prettyExprIr ===")
    let varA = Ir.Var(1, 0, "A")
    let varB = Ir.Var(1, 5, "B")
    let constThree = Ir.Const(1, 0, 3.0)
    check "prettyExprIr Var"    "A"       (prettyExprIr varA)
    check "prettyExprIr Const"  "3"       (prettyExprIr constThree)
    check "prettyExprIr BinOp"  "(A + B)" (prettyExprIr (Ir.BinOp(1, 0, "+", varA, varB)))
    check "prettyExprIr Neg"    "(-A)"    (prettyExprIr (Ir.Neg(1, 0, varA)))
    let applyExpr = Ir.Apply(1, 0, varA, [ Ir.IndexExpr(1, 0, constThree) ])
    check "prettyExprIr Apply"  "A(3)"    (prettyExprIr applyExpr)
    check "prettyExprIr FuncHandle" "@foo" (prettyExprIr (Ir.FuncHandle(1, 0, "foo")))

    Console.WriteLine("=== Phase 2: AnalysisContext + snapshotScope ===")
    let ctx = AnalysisContext()
    ctx.cst.scalarBindings.["n"] <- 5
    ctx.cst.scalarBindings.["m"] <- 3
    // Run snapshot scope -- changes inside should be reverted
    let resultInside =
        ctx.SnapshotScope(fun () ->
            ctx.cst.scalarBindings.["n"] <- 99
            ctx.cst.scalarBindings.["k"] <- 42
            ctx.cst.scalarBindings.["n"])  // returns 99 inside
    check "snapshotScope inside n"    "99" (string resultInside)
    check "snapshotScope restored n"  "5"  (string ctx.cst.scalarBindings.["n"])
    check "snapshotScope restored m"  "3"  (string ctx.cst.scalarBindings.["m"])
    checkBool "snapshotScope k removed" false (ctx.cst.scalarBindings.ContainsKey("k"))

    // Verify exception safety of snapshotScope
    ctx.cst.scalarBindings.["n"] <- 5
    try
        ctx.SnapshotScope(fun () ->
            ctx.cst.scalarBindings.["n"] <- 77
            raise (System.Exception("test exception"))) |> ignore
    with _ -> ()
    check "snapshotScope restored after exception" "5" (string ctx.cst.scalarBindings.["n"])

    Console.WriteLine("=== Phase 2: PathConstraintStack ===")
    let stack = PathConstraints.PathConstraintStack()
    let condExpr = Ir.BinOp(10, 0, ">", Ir.Var(10, 0, "n"), Ir.Const(10, 0, 3.0))
    stack.Push(condExpr, true, 10)
    let snap = stack.Snapshot()
    check "PathConstraints push+snapshot count" "1" (string snap.Length)
    check "PathConstraints description" "n > 3" snap.[0].description
    checkBool "PathConstraints branchTaken" true snap.[0].branchTaken
    stack.Pop()
    check "PathConstraints after pop" "0" (string (stack.Snapshot().Length))
    // Pop on empty is no-op
    stack.Pop()
    check "PathConstraints double pop safe" "0" (string (stack.Snapshot().Length))

    Console.WriteLine("")
    if failures = 0 then
        Console.WriteLine("All Phase 2 smoke tests PASSED")
        0
    else
        Console.Error.WriteLine(string failures + " Phase 2 smoke test(s) FAILED")
        1

[<EntryPoint>]
let main argv =
    if argv.Length >= 1 && argv.[0] = "--test-shapes" then
        runShapesTest ()
    elif argv.Length >= 1 && argv.[0] = "--test-phase2" then
        runPhase2Test ()
    elif argv.Length < 1 then
        Console.Error.WriteLine("Usage: conformal-parse <file.m>")
        1
    else
        let filePath = argv.[0]
        if not (File.Exists filePath) then
            Console.Error.WriteLine("File not found: " + filePath)
            1
        else
            try
                let src = File.ReadAllText(filePath)
                let program = Parser.parseMATLAB src
                let json = Json.programToJson program
                Console.WriteLine(json)
                0
            with
            | Parser.ParseError msg ->
                Console.Error.WriteLine("ParseError: " + msg)
                2
            | Lexer.LexError msg ->
                Console.Error.WriteLine("LexError: " + msg)
                2
            | ex ->
                Console.Error.WriteLine("Error: " + ex.Message)
                3
