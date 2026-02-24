module Program

open System
open System.IO
open Shapes
open SymDim
open Diagnostics
open Builtins
open Context
open Intervals
open SharedTypes
open EvalExpr
open EvalBuiltins
open Analysis

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
    let leftE  = Ir.Var(Ir.loc 1 0, "A")
    let rightE = Ir.Var(Ir.loc 1 5, "B")
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
    let varA = Ir.Var(Ir.loc 1 0, "A")
    let varB = Ir.Var(Ir.loc 1 5, "B")
    let constThree = Ir.Const(Ir.loc 1 0, 3.0)
    check "prettyExprIr Var"    "A"       (prettyExprIr varA)
    check "prettyExprIr Const"  "3"       (prettyExprIr constThree)
    check "prettyExprIr BinOp"  "(A + B)" (prettyExprIr (Ir.BinOp(Ir.loc 1 0, "+", varA, varB)))
    check "prettyExprIr Neg"    "(-A)"    (prettyExprIr (Ir.Neg(Ir.loc 1 0, varA)))
    let applyExpr = Ir.Apply(Ir.loc 1 0, varA, [ Ir.IndexExpr(Ir.loc 1 0, constThree) ])
    check "prettyExprIr Apply"  "A(3)"    (prettyExprIr applyExpr)
    check "prettyExprIr FuncHandle" "@foo" (prettyExprIr (Ir.FuncHandle(Ir.loc 1 0, "foo")))

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
    let condExpr = Ir.BinOp(Ir.loc 10 0, ">", Ir.Var(Ir.loc 10 0, "n"), Ir.Const(Ir.loc 10 0, 3.0))
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

// ---------------------------------------------------------------------------
// Smoke test for Phase 3: AnalysisCore + DimExtract + Intervals + Constraints
//                         + MatrixLiterals + EvalBinop + EvalExpr
// ---------------------------------------------------------------------------

let runPhase3Test () : int =
    let mutable failures = 0

    let check (label: string) (expected: string) (actual: string) =
        if expected = actual then
            Console.WriteLine("  PASS  " + label + " => " + actual)
        else
            Console.Error.WriteLine("  FAIL  " + label + ": expected \"" + expected + "\" got \"" + actual + "\"")
            failures <- failures + 1

    let checkBool (label: string) (expected: bool) (actual: bool) =
        check label (string expected) (string actual)

    // Stub builtinDispatch: returns unknown for all builtins
    let stubDispatch : BuiltinDispatch =
        fun _fname _line _base _args _env _warnings _ctx -> UnknownShape

    Console.WriteLine("=== Phase 3: DimExtract ===")
    let env0 = Env.Env.create ()
    let constExpr5  = Ir.Const(Ir.loc 1 0, 5.0)
    let varNExpr    = Ir.Var(Ir.loc 1 0, "n")

    let dim5 = DimExtract.exprToDimIr constExpr5 env0
    check "exprToDimIr Const(5)" "Concrete 5"
        (match dim5 with Concrete n -> "Concrete " + string n | Symbolic _ -> "Symbolic" | Unknown -> "Unknown")

    let dimN = DimExtract.exprToDimIr varNExpr env0
    check "exprToDimIr Var(n)" "Symbolic n"
        (match dimN with Concrete n -> "Concrete " + string n | Symbolic s -> "Symbolic " + SymDim.toString s | Unknown -> "Unknown")

    Console.WriteLine("=== Phase 3: Interval arithmetic ===")
    let iv13 = Some { lo = Finite 1; hi = Finite 3 }
    let iv25 = Some { lo = Finite 2; hi = Finite 5 }
    let iv_neg = Some { lo = Finite -5; hi = Finite -1 }

    let boundStr b = match b with Finite n -> string n | Unbounded -> "inf" | SymBound _ -> "sym"
    let ivStr iv = match iv with Some i -> "[" + boundStr i.lo + "," + boundStr i.hi + "]" | None -> "None"

    let ivAdd = intervalAdd iv13 iv25
    check "intervalAdd [1,3]+[2,5]" "[3,8]" (ivStr ivAdd)

    let ivSub = intervalSub iv25 iv13
    check "intervalSub [2,5]-[1,3]" "[-1,4]" (ivStr ivSub)

    let ivMul = intervalMul iv13 iv13
    check "intervalMul [1,3]*[1,3]" "[1,9]" (ivStr ivMul)

    let ivJoin = joinInterval iv13 iv25
    check "joinInterval [1,3] [2,5]" "[1,5]" (ivStr ivJoin)

    let ivMeet = meetInterval { lo = Finite 1; hi = Finite 5 } { lo = Finite 3; hi = Finite 8 }
    check "meetInterval [1,5] [3,8]" "Some [3,5]"
        (match ivMeet with Some i -> "Some [" + boundStr i.lo + "," + boundStr i.hi + "]" | None -> "None")

    checkBool "intervalDefinitelyNegative neg"      true  (intervalDefinitelyNegative  iv_neg)
    checkBool "intervalDefinitelyNegative pos"      false (intervalDefinitelyNegative  iv25)
    checkBool "intervalDefinitelyPositive pos"      true  (intervalDefinitelyPositive  iv25)
    checkBool "intervalDefinitelyPositive neg"      false (intervalDefinitelyPositive  iv_neg)
    checkBool "intervalIsExactlyZero zero"          true  (intervalIsExactlyZero (Some { lo = Finite 0; hi = Finite 0 }))
    checkBool "intervalIsExactlyZero nonzero"       false (intervalIsExactlyZero iv13)

    Console.WriteLine("=== Phase 3: EvalBinop ===")
    let ctx0 = AnalysisContext()
    let env1 = Env.Env.create ()
    let warnRef = ResizeArray<Diagnostic>()
    let dummyExpr = Ir.Const(Ir.loc 1 0, 1.0)
    let noIv _ = None

    // matmul 3x4 * 4x2 -> 3x2
    let leftM  = Matrix(Concrete 3, Concrete 4)
    let rightM = Matrix(Concrete 4, Concrete 2)
    let result = EvalBinop.evalBinopIr "*" leftM rightM warnRef dummyExpr dummyExpr 1 ctx0 env1 noIv
    check "matmul 3x4 * 4x2 -> 3x2" "matrix[3 x 2]" (shapeToString result)
    checkBool "matmul no warnings" true (warnRef.Count = 0)

    warnRef.Clear()
    // matmul dimension mismatch: 3x4 * 5x2 should produce unknown + warning
    let rightMbad = Matrix(Concrete 5, Concrete 2)
    let resultBad = EvalBinop.evalBinopIr "*" leftM rightMbad warnRef dummyExpr dummyExpr 1 ctx0 env1 noIv
    check "matmul mismatch -> unknown" "unknown" (shapeToString resultBad)
    checkBool "matmul mismatch has warning" true (warnRef |> Seq.exists (fun d -> d.code = "W_INNER_DIM_MISMATCH"))

    warnRef.Clear()
    // elementwise: 3x4 .* 3x4 -> 3x4
    let ewResult = EvalBinop.evalBinopIr ".*" leftM leftM warnRef dummyExpr dummyExpr 1 ctx0 env1 noIv
    check "elementwise 3x4 .* 3x4 -> 3x4" "matrix[3 x 4]" (shapeToString ewResult)

    Console.WriteLine("=== Phase 3: MatrixLiterals ===")
    let warnRef2 = ResizeArray<Diagnostic>()
    let ctx1 = AnalysisContext()
    let env2 = Env.Env.create ()

    // Empty literal []
    let emptyShape = MatrixLiterals.inferMatrixLiteralShape [] 1 warnRef2 ctx1 env2
    check "empty literal -> matrix[0 x 0]" "matrix[0 x 0]" (shapeToString emptyShape)

    // Row vector [a b]: 1 row of [scalar, scalar] -> matrix[1 x 2]
    let rowShape = MatrixLiterals.inferMatrixLiteralShape [[Scalar; Scalar]] 1 warnRef2 ctx1 env2
    check "row [scalar scalar] -> matrix[1 x 2]" "matrix[1 x 2]" (shapeToString rowShape)

    // Column vector [a; b]: 2 rows of [scalar] each -> matrix[2 x 1]
    let colShape = MatrixLiterals.inferMatrixLiteralShape [[Scalar]; [Scalar]] 1 warnRef2 ctx1 env2
    check "col [scalar; scalar] -> matrix[2 x 1]" "matrix[2 x 1]" (shapeToString colShape)

    Console.WriteLine("=== Phase 3: evalExprIr ===")
    let warnRef3 = ResizeArray<Diagnostic>()
    let ctx2 = AnalysisContext()
    let env3 = Env.Env.create ()

    // Const -> Scalar
    let constResult = evalExprIr (Ir.Const(Ir.loc 1 0, 3.14)) env3 warnRef3 ctx2 None stubDispatch
    check "evalExprIr Const -> scalar" "scalar" (shapeToString constResult)

    // Var bound to matrix
    Env.Env.set env3 "A" (Matrix(Concrete 3, Concrete 4))
    let varResult = evalExprIr (Ir.Var(Ir.loc 1 0, "A")) env3 warnRef3 ctx2 None stubDispatch
    check "evalExprIr Var(A) -> matrix[3 x 4]" "matrix[3 x 4]" (shapeToString varResult)

    // Var unbound -> unknown
    let varUndef = evalExprIr (Ir.Var(Ir.loc 1 0, "undefined_var")) env3 warnRef3 ctx2 None stubDispatch
    check "evalExprIr unbound Var -> unknown" "unknown" (shapeToString varUndef)

    // MATLAB constant pi -> scalar
    let piResult = evalExprIr (Ir.Var(Ir.loc 1 0, "pi")) env3 warnRef3 ctx2 None stubDispatch
    check "evalExprIr pi -> scalar" "scalar" (shapeToString piResult)

    // StringLit -> string
    let strResult = evalExprIr (Ir.StringLit(Ir.loc 1 0, "hello")) env3 warnRef3 ctx2 None stubDispatch
    check "evalExprIr StringLit -> string" "string" (shapeToString strResult)

    Console.WriteLine("")
    if failures = 0 then
        Console.WriteLine("All Phase 3 smoke tests PASSED")
        0
    else
        Console.Error.WriteLine(string failures + " Phase 3 smoke test(s) FAILED")
        1


// ---------------------------------------------------------------------------
// Smoke test for Phase 4: EvalBuiltins + StmtFuncAnalysis + Analysis
// ---------------------------------------------------------------------------

let runPhase4Test () : int =
    let mutable failures = 0

    let check (label: string) (expected: string) (actual: string) =
        if expected = actual then
            Console.WriteLine("  PASS  " + label + " => " + actual)
        else
            Console.Error.WriteLine("  FAIL  " + label + ": expected \"" + expected + "\" got \"" + actual + "\"")
            failures <- failures + 1

    let checkBool (label: string) (expected: bool) (actual: bool) =
        check label (string expected) (string actual)

    // --- EvalBuiltins: builtin dispatch sets ---
    Console.WriteLine("=== Phase 4: EvalBuiltins dispatch sets ===")
    checkBool "zeros in PASSTHROUGH"         false (Set.contains "zeros"  PASSTHROUGH_BUILTINS)
    checkBool "abs in PASSTHROUGH"           true  (Set.contains "abs"    PASSTHROUGH_BUILTINS)
    checkBool "sum in REDUCTION"             true  (Set.contains "sum"    REDUCTION_BUILTINS)
    checkBool "isscalar in SCALAR_PREDICATE" true  (Set.contains "isscalar" SCALAR_PREDICATE_BUILTINS)
    checkBool "double in TYPE_CAST"          true  (Set.contains "double"  TYPE_CAST_BUILTINS)
    checkBool "length in SCALAR_QUERY"       true  (Set.contains "length"  SCALAR_QUERY_BUILTINS)
    checkBool "eye in MATRIX_CONSTRUCTOR"     true  (Set.contains "eye"     MATRIX_CONSTRUCTOR_BUILTINS)

    // --- EvalBuiltins: zeros(3,4) -> matrix[3 x 4] ---
    Console.WriteLine("=== Phase 4: evalBuiltinCall zeros(3,4) ===")
    let env0 = Env.Env.create ()
    let warnRef = ResizeArray<Diagnostic>()
    let ctx0 = AnalysisContext()

    let stubEval (e: Ir.Expr) (_env: Env.Env) (_w: ResizeArray<Diagnostic>) (_ctx: AnalysisContext) : Shape =
        match e with
        | Ir.Const(_, v) -> Scalar
        | _ -> UnknownShape

    let stubGetInterval (_e: Ir.Expr) (_env: Env.Env) (_ctx: AnalysisContext) : Interval option =
        None

    let zerosArgs = [ Ir.IndexExpr(Ir.loc 1 0, Ir.Const(Ir.loc 1 0, 3.0)); Ir.IndexExpr(Ir.loc 1 0, Ir.Const(Ir.loc 1 0, 4.0)) ]
    let zerosResult = evalBuiltinCall "zeros" 1 zerosArgs env0 warnRef ctx0 stubEval stubGetInterval
    check "zeros(3,4)" "matrix[3 x 4]" (shapeToString zerosResult)
    checkBool "zeros(3,4) no warnings" true (warnRef.Count = 0)

    // --- EvalBuiltins: size(A) -> scalar ---
    Console.WriteLine("=== Phase 4: evalBuiltinCall size(A) ===")
    Env.Env.set env0 "A" (Matrix(Concrete 3, Concrete 4))
    let sizeArgs = [ Ir.IndexExpr(Ir.loc 1 0, Ir.Var(Ir.loc 1 0, "A")) ]
    let sizeEval (e: Ir.Expr) (_env: Env.Env) (_w: ResizeArray<Diagnostic>) (_ctx: AnalysisContext) : Shape =
        match e with
        | Ir.Var(_, "A") -> Matrix(Concrete 3, Concrete 4)
        | _ -> UnknownShape
    let sizeResult = evalBuiltinCall "size" 1 sizeArgs env0 warnRef ctx0 sizeEval stubGetInterval
    // size(A) with 1 arg returns matrix[1 x 2]
    check "size(A) 1-arg" "matrix[1 x 2]" (shapeToString sizeResult)

    // --- analyzeProgramIr: simple assignment ---
    Console.WriteLine("=== Phase 4: analyzeProgramIr simple assignment ===")
    let simpleProgram : Ir.Program = {
        body = [ Ir.Assign(Ir.loc 1 0, "x", Ir.Const(Ir.loc 1 0, 5.0)) ]
    }
    let ctx1 = AnalysisContext()
    let (env1, diags1) = analyzeProgramIr simpleProgram ctx1
    check "x = 5 -> scalar" "scalar" (shapeToString (Env.Env.get env1 "x"))
    checkBool "no diagnostics" true diags1.IsEmpty

    // --- analyzeProgramIr: matrix assignment ---
    Console.WriteLine("=== Phase 4: analyzeProgramIr zeros builtin ===")
    let zerosProgram : Ir.Program = {
        body = [
            Ir.Assign(Ir.loc 1 0, "A",
                Ir.Apply(Ir.loc 1 0,
                    Ir.Var(Ir.loc 1 0, "zeros"),
                    [ Ir.IndexExpr(Ir.loc 1 0, Ir.Const(Ir.loc 1 0, 3.0))
                      Ir.IndexExpr(Ir.loc 1 0, Ir.Const(Ir.loc 1 0, 4.0)) ]))
        ]
    }
    let ctx2 = AnalysisContext()
    let (env2, diags2) = analyzeProgramIr zerosProgram ctx2
    check "A = zeros(3,4) -> matrix[3 x 4]" "matrix[3 x 4]" (shapeToString (Env.Env.get env2 "A"))

    // --- analyzeProgramIr: function definition + call ---
    Console.WriteLine("=== Phase 4: analyzeProgramIr function call ===")
    let funcProgram : Ir.Program = {
        body = [
            Ir.FunctionDef(Ir.loc 1 0, "makeVec",
                ["n"],
                ["v"],
                [ Ir.Assign(Ir.loc 2 0, "v",
                      Ir.Apply(Ir.loc 2 0, Ir.Var(Ir.loc 2 0, "zeros"),
                          [ Ir.IndexExpr(Ir.loc 2 0, Ir.Var(Ir.loc 2 0, "n"))
                            Ir.IndexExpr(Ir.loc 2 0, Ir.Const(Ir.loc 2 0, 1.0)) ])) ])
            Ir.Assign(Ir.loc 5 0, "result",
                Ir.Apply(Ir.loc 5 0, Ir.Var(Ir.loc 5 0, "makeVec"),
                    [ Ir.IndexExpr(Ir.loc 5 0, Ir.Const(Ir.loc 5 0, 5.0)) ]))
        ]
    }
    let ctx3 = AnalysisContext()
    let (env3, _diags3) = analyzeProgramIr funcProgram ctx3
    check "makeVec(5) -> matrix[5 x 1]" "matrix[5 x 1]" (shapeToString (Env.Env.get env3 "result"))

    // --- MULTI_SUPPORTED_FORMS exists ---
    Console.WriteLine("=== Phase 4: MULTI_SUPPORTED_FORMS ===")
    checkBool "size in MULTI_SUPPORTED_FORMS" true (Map.containsKey "size" MULTI_SUPPORTED_FORMS)
    checkBool "eig in MULTI_SUPPORTED_FORMS"  true (Map.containsKey "eig"  MULTI_SUPPORTED_FORMS)

    Console.WriteLine("")
    if failures = 0 then
        Console.WriteLine("All Phase 4 smoke tests PASSED")
        0
    else
        Console.Error.WriteLine(string failures + " Phase 4 smoke test(s) FAILED")
        1


[<EntryPoint>]
let main argv =
    // Smoke tests handled here (Program.fs is last in compilation order)
    if argv.Length >= 1 && argv.[0] = "--test-shapes" then
        runShapesTest ()
    elif argv.Length >= 1 && argv.[0] = "--test-phase2" then
        runPhase2Test ()
    elif argv.Length >= 1 && argv.[0] = "--test-phase3" then
        runPhase3Test ()
    elif argv.Length >= 1 && argv.[0] = "--test-phase4" then
        runPhase4Test ()
    elif argv |> Array.contains "--lsp" then
        // Start F# LSP server (JSON-RPC over stdio)
        LspServer.startLsp ()
    else
        // All other dispatch goes through Cli.run
        Cli.run argv
