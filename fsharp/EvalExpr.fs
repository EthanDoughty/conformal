module EvalExpr

open Ir
open Shapes
open Env
open Context
open Diagnostics
open Builtins
open EndHelpers
open DimExtract

// ---------------------------------------------------------------------------
// EvalExpr: core expression evaluator.
// Port of analysis/eval_expr.py
//
// CIRCULAR DEPENDENCY NOTE:
//   EvalExpr depends on EvalBuiltins for builtin dispatch, but EvalBuiltins
//   must come AFTER EvalExpr in the F# compilation order.
//   Solution: the main evaluator takes a BuiltinDispatch callback parameter
//   that Phase 4 wires to EvalBuiltins.evalBuiltinCall.
// ---------------------------------------------------------------------------

/// BuiltinDispatch: callback type for builtin function dispatch.
/// Signature: fname -> line -> baseExpr -> args -> env -> warnings ref -> ctx -> Shape
type BuiltinDispatch = string -> int -> Expr -> IndexArg list -> Env -> Diagnostics.Diagnostic list ref -> AnalysisContext -> Shape

// ---------------------------------------------------------------------------
// MATLAB predefined constants
// ---------------------------------------------------------------------------

let private matlabConstants : Map<string, Shape> =
    Map.ofList [
        "pi", Scalar; "inf", Scalar; "Inf", Scalar
        "nan", Scalar; "NaN", Scalar; "eps", Scalar
        "true", Scalar; "false", Scalar
        "i", Scalar; "j", Scalar
        "realmin", Scalar; "realmax", Scalar; "intmax", Scalar
        "intmin", Scalar; "flintmax", Scalar
    ]


// ---------------------------------------------------------------------------
// Interval evaluator (parallel to shape inference, no shape side effects)
// ---------------------------------------------------------------------------

/// getExprInterval: compute integer interval for an expression.
let rec getExprInterval (expr: Expr) (env: Env) (ctx: AnalysisContext) : Intervals.Interval option =
    match expr with
    | Const(_, _, v) ->
        // Integer constants only
        if v = System.Math.Floor v && not (System.Double.IsInfinity v) then
            let iv = int v
            Some { Intervals.lo = Intervals.Finite iv; Intervals.hi = Intervals.Finite iv }
        else None
    | Var(_, _, name) ->
        match ctx.cst.valueRanges.TryGetValue(name) with
        | true, (:? Intervals.Interval as iv) -> Some iv
        | _ -> None
    | Neg(_, _, operand) ->
        let operandIv = getExprInterval operand env ctx
        Intervals.intervalNeg operandIv
    | BinOp(_, _, op, left, right) ->
        let lIv = getExprInterval left env ctx
        let rIv = getExprInterval right env ctx
        match op with
        | "+" -> Intervals.intervalAdd lIv rIv
        | "-" -> Intervals.intervalSub lIv rIv
        | "*" -> Intervals.intervalMul lIv rIv
        | "/" -> None   // division interval not implemented
        | "^" | ".^" ->
            match lIv, rIv with
            | Some la, Some ra ->
                match la.lo, la.hi, ra.lo, ra.hi with
                | Intervals.Finite ll, Intervals.Finite lh, Intervals.Finite rl, Intervals.Finite rh
                    when rl = rh && rl >= 0 ->
                    let exp = rl
                    let corners = [ pown ll exp; pown lh exp ]
                    let corners =
                        if ll < 0 && 0 < lh && exp % 2 = 0 then corners @ [0]
                        else corners
                    try
                        Some { Intervals.lo = Intervals.Finite (List.min corners)
                               Intervals.hi = Intervals.Finite (List.max corners) }
                    with _ -> None
                | _ -> None
            | _ -> None
        | _ -> None
    | _ -> None


/// getConcreteDimSize: get concrete dimension size from int or SymDim via interval lookup.
let getConcreteDimSize (dim: Dim) (ctx: AnalysisContext) : int option =
    match dim with
    | Concrete n -> Some n
    | Symbolic s ->
        // Only extract from simple single-variable SymDims (e.g., 'n' but not '2*n' or 'n+1')
        match s._terms with
        | [([varName, 1], coeff)] when coeff = SymDim.Rational.One ->
            match ctx.cst.valueRanges.TryGetValue(varName) with
            | true, (:? Intervals.Interval as iv) ->
                match iv.lo, iv.hi with
                | Intervals.Finite lo, Intervals.Finite hi when lo = hi -> Some lo
                | _ -> None
            | _ -> None
        | _ -> None
    | Unknown -> None


// ---------------------------------------------------------------------------
// Main expression evaluator
// ---------------------------------------------------------------------------

/// joinAllElements: join all element shapes from a cell element tracking map.
let private joinAllElements (elemMap: Map<int, Shape>) : Shape =
    if elemMap.IsEmpty then UnknownShape
    else
        Map.fold (fun acc _ shp -> joinShape acc shp) Bottom elemMap


/// evalIndexArgToShape: evaluate an IndexArg to a Shape.
let evalIndexArgToShape
    (arg: IndexArg)
    (env: Env)
    (warnings: Diagnostic list ref)
    (ctx: AnalysisContext)
    (containerShape: Shape option)
    (evalFn: Expr -> Env -> Diagnostic list ref -> AnalysisContext -> Shape option -> Shape)
    : Shape =
    match arg with
    | IndexExpr(_, _, expr) -> evalFn expr env warnings ctx containerShape
    | Range(_, _, startExpr, endExpr) ->
        // Evaluate start/end for side effects
        evalFn startExpr env warnings ctx containerShape |> ignore
        evalFn endExpr   env warnings ctx containerShape |> ignore
        Matrix(Concrete 1, Unknown)   // 1 x unknown
    | Colon _ -> UnknownShape


/// indexArgToExtentIr: return how many rows/cols this IndexArg selects.
let indexArgToExtentIr
    (arg: IndexArg)
    (env: Env)
    (warnings: Diagnostic list ref)
    (line: int)
    (ctx: AnalysisContext)
    (containerShape: Shape option)
    (dimSize: Dim)
    (evalFn: Expr -> Env -> Diagnostic list ref -> AnalysisContext -> Shape option -> Shape)
    : Dim =
    match arg with
    | Colon _ -> Unknown

    | Range(_, _, startExpr, endExpr) ->
        let startShape = evalFn startExpr env warnings ctx containerShape
        let endShape   = evalFn endExpr   env warnings ctx containerShape

        match startShape, endShape with
        | Matrix _, _ | _, Matrix _ ->
            warnings.Value <- warnings.Value @
                [ warnRangeEndpointsMustBeScalar line arg startShape endShape ]
            Unknown
        | _ ->
            // Try to extract dimension values from endpoints
            let mutable a = exprToDimIr startExpr env
            let mutable b = exprToDimIr endExpr env

            // If endpoints need End substitution
            if a = Unknown then a <- exprToDimWithEnd startExpr env dimSize
            if b = Unknown then b <- exprToDimWithEnd endExpr env dimSize

            match a, b with
            | Unknown, _ | _, Unknown -> Unknown
            | Concrete a', Concrete b' ->
                if b' < a' then
                    warnings.Value <- warnings.Value @
                        [ warnInvalidRangeEndLtStart line arg ]
                    Unknown
                else Concrete ((b' - a') + 1)
            | _ -> addDim (subDim b a) (Concrete 1)   // (b-a)+1 symbolic

    | IndexExpr(_, _, expr) ->
        let s = evalFn expr env warnings ctx containerShape
        match s with
        | Matrix _ ->
            match expr with
            | MatrixLit _ -> Unknown   // matrix literal as subscript: valid MATLAB
            | _ ->
                warnings.Value <- warnings.Value @
                    [ warnNonScalarIndexArg line arg s ]
                Unknown
        | _ -> Concrete 1


/// evalExprIr: main expression evaluator.
/// containerShape: optional shape of container being indexed (for End keyword resolution).
/// builtinDispatch: callback for builtin function dispatch (set to stub in Phase 3).
let rec evalExprIr
    (expr: Expr)
    (env: Env)
    (warnings: Diagnostic list ref)
    (ctx: AnalysisContext)
    (containerShape: Shape option)
    (builtinDispatch: BuiltinDispatch)
    : Shape =

    match expr with
    // ------------------------------------------------------------------
    // Variables / constants
    // ------------------------------------------------------------------
    | Var(_, _, name) ->
        let shape = Env.get env name
        if shape = Bottom then
            match Map.tryFind name matlabConstants with
            | Some s -> s
            | None   -> UnknownShape
        else shape

    | Const _ -> Scalar

    | StringLit _ -> StringShape

    | End(line, _) ->
        match containerShape with
        | None ->
            warnings.Value <- warnings.Value @ [ warnEndOutsideIndexing line ]
            UnknownShape
        | Some _ -> Scalar   // End resolves to scalar index

    // ------------------------------------------------------------------
    // Matrix literal
    // ------------------------------------------------------------------
    | MatrixLit(line, _, rows) ->
        let shapeRows =
            rows |> List.map (fun row ->
                row |> List.map (fun e -> evalExprIr e env warnings ctx None builtinDispatch))
        let warningsRef = warnings
        MatrixLiterals.inferMatrixLiteralShape shapeRows line warningsRef ctx env

    // ------------------------------------------------------------------
    // Cell literal
    // ------------------------------------------------------------------
    | CellLit(_, _, rows) ->
        if rows.IsEmpty then
            Cell(Concrete 0, Concrete 0, Some Map.empty)
        else
            let numRows = rows.Length
            let numCols = if rows.IsEmpty then 0 else rows.[0].Length

            // Check rectangular
            let isRagged = rows |> List.exists (fun row -> row.Length <> numCols)
            if isRagged then
                Cell(Unknown, Unknown, None)
            else
                // Column-major indexing
                let elemShapes = System.Collections.Generic.Dictionary<int, Shape>()
                for rowIdx, row in rows |> List.mapi (fun i r -> (i, r)) do
                    for colIdx, e in row |> List.mapi (fun i x -> (i, x)) do
                        let elemShape = evalExprIr e env warnings ctx None builtinDispatch
                        let linearIdx = colIdx * numRows + rowIdx
                        if not (isUnknown elemShape) then
                            elemShapes.[linearIdx] <- elemShape
                let elemMap =
                    elemShapes |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
                Cell(Concrete numRows, Concrete numCols, Some elemMap)

    // ------------------------------------------------------------------
    // Curly indexing: c{i} or c{i,j}
    // ------------------------------------------------------------------
    | CurlyApply(line, _, baseExpr, args) ->
        let baseShape = evalExprIr baseExpr env warnings ctx None builtinDispatch

        if not (isCell baseShape) then
            for arg in args do
                evalIndexArgToShape arg env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch)
                |> ignore
            if not (isUnknown baseShape) && not (isEmptyMatrix baseShape) then
                warnings.Value <- warnings.Value @
                    [ warnCurlyIndexingNonCell line baseShape ]
            UnknownShape
        else
            match baseShape with
            | Cell(_, _, None) ->
                for arg in args do
                    evalIndexArgToShape arg env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch)
                    |> ignore
                UnknownShape
            | Cell(rows, cols, Some elemMap) ->
                if args.Length = 1 then
                    let arg = args.[0]
                    match arg with
                    | IndexExpr(_, _, Const(_, _, v)) ->
                        // Literal 1D index
                        let idx0 = int v - 1
                        match Map.tryFind idx0 elemMap with
                        | Some s -> s
                        | None   -> UnknownShape
                    | IndexExpr(_, _, End _) ->
                        match rows, cols with
                        | Concrete nr, Concrete nc ->
                            let total = nr * nc
                            match Map.tryFind (total - 1) elemMap with
                            | Some s -> s
                            | None   -> UnknownShape
                        | _ -> joinAllElements elemMap
                    | IndexExpr(_, _, (BinOp(_, _, _, _, _) as binOpExpr))
                        when binopContainsEnd binOpExpr ->
                        match rows, cols with
                        | Concrete nr, Concrete nc ->
                            let total = nr * nc
                            let idxOpt = evalEndArithmetic binOpExpr total
                            match idxOpt with
                            | Some idx1 when idx1 >= 1 && idx1 <= total ->
                                match Map.tryFind (idx1 - 1) elemMap with
                                | Some s -> s
                                | None   -> UnknownShape
                            | _ -> joinAllElements elemMap
                        | _ -> joinAllElements elemMap
                    | Colon _ -> joinAllElements elemMap
                    | Range _ ->
                        evalIndexArgToShape arg env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch)
                        |> ignore
                        joinAllElements elemMap
                    | _ ->
                        evalIndexArgToShape arg env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch)
                        |> ignore
                        joinAllElements elemMap

                elif args.Length = 2 then
                    let argRow = args.[0]
                    let argCol = args.[1]

                    let tryGetRowIdx () =
                        match argRow with
                        | IndexExpr(_, _, Const(_, _, v)) -> Some (int v - 1)
                        | IndexExpr(_, _, End _) ->
                            match rows with Concrete nr -> Some (nr - 1) | _ -> None
                        | _ -> None

                    let tryGetColIdx () =
                        match argCol with
                        | IndexExpr(_, _, Const(_, _, v)) -> Some (int v - 1)
                        | IndexExpr(_, _, End _) ->
                            match cols with Concrete nc -> Some (nc - 1) | _ -> None
                        | _ -> None

                    match tryGetRowIdx (), tryGetColIdx (), rows with
                    | Some ri, Some ci, Concrete nr ->
                        let linearIdx = ci * nr + ri
                        match Map.tryFind linearIdx elemMap with
                        | Some s -> s
                        | None   -> UnknownShape
                    | _ ->
                        evalIndexArgToShape argRow env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch) |> ignore
                        evalIndexArgToShape argCol env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch) |> ignore
                        joinAllElements elemMap

                else
                    for arg in args do
                        evalIndexArgToShape arg env warnings ctx (Some baseShape) (fun e en w c cs -> evalExprIr e en w c cs builtinDispatch) |> ignore
                    UnknownShape
            | _ -> UnknownShape

    // ------------------------------------------------------------------
    // Apply: function call or array indexing
    // ------------------------------------------------------------------
    | Apply(line, _, baseExpr, args) ->
        evalApply line baseExpr args env warnings ctx builtinDispatch

    // ------------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------------
    | Transpose(line, _, operand) ->
        let inner = evalExprIr operand env warnings ctx None builtinDispatch
        if not (isNumeric inner) && not (isUnknown inner) then
            warnings.Value <- warnings.Value @ [ warnTransposeTypeMismatch line inner ]
            UnknownShape
        else
            match inner with
            | Matrix(r, c) -> Matrix(c, r)
            | _ -> inner

    // ------------------------------------------------------------------
    // Negation / logical Not
    // ------------------------------------------------------------------
    | Neg(line, _, operand) ->
        let s = evalExprIr operand env warnings ctx None builtinDispatch
        if not (isNumeric s) && not (isUnknown s) then
            warnings.Value <- warnings.Value @ [ warnNegateTypeMismatch line s ]
            UnknownShape
        else s

    | Not(line, _, operand) ->
        let s = evalExprIr operand env warnings ctx None builtinDispatch
        if not (isNumeric s) && not (isUnknown s) then
            warnings.Value <- warnings.Value @ [ warnNotTypeMismatch line s ]
            UnknownShape
        else s

    // ------------------------------------------------------------------
    // Field access: s.field
    // ------------------------------------------------------------------
    | FieldAccess(line, _, baseExpr, field) ->
        let baseShape = evalExprIr baseExpr env warnings ctx None builtinDispatch
        if field = "<dynamic>" then
            UnknownShape  // dynamic field access: conservative
        else
            match baseShape with
            | Struct(fields, isOpen) ->
                let fmap = Map.ofList fields
                match Map.tryFind field fmap with
                | None ->
                    if isOpen then UnknownShape  // open struct: might exist
                    else
                        warnings.Value <- warnings.Value @
                            [ warnStructFieldNotFound line field baseShape ]
                        UnknownShape
                | Some fs ->
                    if fs = Bottom then UnknownShape else fs
            | UnknownShape -> UnknownShape
            | _ ->
                if not (isEmptyMatrix baseShape) then
                    warnings.Value <- warnings.Value @
                        [ warnFieldAccessNonStruct line baseShape ]
                UnknownShape

    // ------------------------------------------------------------------
    // Lambda: @(params) body
    // ------------------------------------------------------------------
    | Lambda(_, _, parms, body) ->
        let lambdaId = ctx.call.nextLambdaId
        ctx.call.nextLambdaId <- lambdaId + 1
        // Store snapshot of current env as closure
        ctx.call.lambdaMetadata.[lambdaId] <- box (parms, body, Env.copy env)
        FunctionHandle(Some (Set.singleton lambdaId))

    // ------------------------------------------------------------------
    // Function handle: @funcName
    // ------------------------------------------------------------------
    | FuncHandle(line, _, name) ->
        let handleId = ctx.call.nextLambdaId
        ctx.call.nextLambdaId <- handleId + 1
        if ctx.call.functionRegistry.ContainsKey(name) ||
           Set.contains name KNOWN_BUILTINS ||
           ctx.ws.externalFunctions.ContainsKey(name) ||
           ctx.call.nestedFunctionRegistry.ContainsKey(name) then
            ctx.call.handleRegistry.[handleId] <- name
            FunctionHandle(Some (Set.singleton handleId))
        else
            warnings.Value <- warnings.Value @ [ warnUnknownFunction line name ]
            FunctionHandle None   // opaque

    // ------------------------------------------------------------------
    // BinOp
    // ------------------------------------------------------------------
    | BinOp(line, _, op, left, right) ->
        let leftShape  = evalExprIr left  env warnings ctx containerShape builtinDispatch
        let rightShape = evalExprIr right env warnings ctx containerShape builtinDispatch
        let getDivisorIv e = getExprInterval e env ctx
        EvalBinop.evalBinopIr op leftShape rightShape warnings left right line ctx env getDivisorIv


/// evalApply: evaluate an Apply node (function call or array indexing).
and private evalApply
    (line: int)
    (baseExpr: Expr)
    (args: IndexArg list)
    (env: Env)
    (warnings: Diagnostic list ref)
    (ctx: AnalysisContext)
    (builtinDispatch: BuiltinDispatch)
    : Shape =

    // Priority 1: base variable is a function handle (shadows builtins)
    match baseExpr with
    | Var(_, _, baseVarName) ->
        let baseVarShape = Env.get env baseVarName
        if isFunctionHandle baseVarShape then
            evalHandleCall baseVarName baseVarShape line args env warnings ctx builtinDispatch
        elif Set.contains baseVarName KNOWN_BUILTINS then
            // Priority 2: known builtin
            builtinDispatch baseVarName line baseExpr args env warnings ctx
        elif not (Env.hasLocal env baseVarName) then
            // Priority 3-5: unbound — user function, nested, external, or unknown
            if ctx.call.functionRegistry.ContainsKey(baseVarName) then
                // STUB for Phase 4: user-defined function call
                UnknownShape
            elif ctx.call.nestedFunctionRegistry.ContainsKey(baseVarName) then
                // STUB for Phase 4: nested function call
                UnknownShape
            elif ctx.ws.externalFunctions.ContainsKey(baseVarName) then
                // STUB for Phase 4: external workspace function call
                UnknownShape
            else
                warnings.Value <- warnings.Value @ [ warnUnknownFunction line baseVarName ]
                UnknownShape
        else
            // Priority 6: bound non-handle variable — treat as indexing
            let baseShape = evalExprIr baseExpr env warnings ctx None builtinDispatch
            evalIndexing baseShape args line env warnings ctx builtinDispatch
    | _ ->
        // Non-variable base: evaluate and treat as indexing
        let baseShape = evalExprIr baseExpr env warnings ctx None builtinDispatch
        evalIndexing baseShape args line env warnings ctx builtinDispatch


/// evalHandleCall: dispatch a call through a function handle variable.
and private evalHandleCall
    (baseVarName: string)
    (baseVarShape: Shape)
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: Diagnostic list ref)
    (ctx: AnalysisContext)
    (builtinDispatch: BuiltinDispatch)
    : Shape =

    match baseVarShape with
    | FunctionHandle None ->
        // Opaque handle (no IDs)
        warnings.Value <- warnings.Value @ [ warnLambdaCallApproximate line baseVarName ]
        UnknownShape
    | FunctionHandle(Some lambdaIds) ->
        // Evaluate argument shapes
        let argShapes =
            args |> List.map (fun arg ->
                match arg with
                | Colon _ | Range _ -> UnknownShape
                | IndexExpr(_, _, e) -> evalExprIr e env warnings ctx None builtinDispatch)

        let results = System.Collections.Generic.List<Shape>()

        for callableId in lambdaIds |> Set.toList |> List.sort do
            match ctx.call.lambdaMetadata.TryGetValue(callableId) with
            | true, (:? (string list * Expr * Env) as lambdaMeta) ->
                let (parms, bodyExpr, closureEnv) = lambdaMeta

                // Cache key
                let argDimAliases =
                    List.mapi2 (fun i param arg ->
                        match arg with
                        | IndexExpr(_, _, e) -> (param, exprToDimIr e env)
                        | _ -> (param, Unknown)) parms args
                    |> List.take (min parms.Length args.Length)

                let cacheKey =
                    "lambda:" + string callableId + ":" +
                    (argShapes |> List.map (fun s -> shapeToString s) |> String.concat ",") + ":" +
                    (argDimAliases |> List.map (fun (p, d) -> p + "=" + dimStr d) |> String.concat ",")

                match ctx.call.analysisCache.TryGetValue(cacheKey) with
                | true, (:? (Shape * Diagnostic list) as cached) ->
                    let (cachedShape, cachedWarnings) = cached
                    warnings.Value <- warnings.Value @ cachedWarnings
                    results.Add(cachedShape)
                | _ ->
                    // Recursion guard
                    if ctx.call.analyzingLambdas.Contains(callableId) then
                        warnings.Value <- warnings.Value @ [ warnRecursiveLambda line ]
                        results.Add(UnknownShape)
                    else
                        // Check arg count
                        if argShapes.Length <> parms.Length then
                            warnings.Value <- warnings.Value @
                                [ warnLambdaArgCountMismatch line parms.Length argShapes.Length ]
                            results.Add(UnknownShape)
                        else
                            ctx.call.analyzingLambdas.Add(callableId) |> ignore
                            try
                                ctx.SnapshotScope(fun () ->
                                    let callEnv = Env.copy closureEnv
                                    // Allow self-reference for recursion detection
                                    Env.set callEnv baseVarName (FunctionHandle(Some (Set.singleton callableId)))
                                    // Bind parameters
                                    for i, (param, argShape) in List.mapi (fun i x -> (i, x)) (List.zip parms argShapes) do
                                        Env.set callEnv param argShape
                                        if i < args.Length then
                                            match args.[i] with
                                            | IndexExpr(_, _, e) ->
                                                let callerDim = exprToDimIr e env
                                                if callerDim <> Unknown then
                                                    callEnv.dimAliases <- Map.add param callerDim callEnv.dimAliases
                                            | _ -> ()

                                    // Analyze body
                                    let lambdaWarnings = ref []
                                    let result = evalExprIr bodyExpr callEnv lambdaWarnings ctx None builtinDispatch

                                    // Cache result
                                    ctx.call.analysisCache.[cacheKey] <- box (result, lambdaWarnings.Value)
                                    warnings.Value <- warnings.Value @ lambdaWarnings.Value
                                    results.Add(result)) |> ignore
                            finally
                                ctx.call.analyzingLambdas.Remove(callableId) |> ignore

            | true, _ ->
                // lambdaMetadata doesn't match expected type (shouldn't happen)
                results.Add(UnknownShape)
            | false, _ ->
                // Not in lambdaMetadata; check handle registry
                match ctx.call.handleRegistry.TryGetValue(callableId) with
                | true, funcName ->
                    if ctx.call.functionRegistry.ContainsKey(funcName) then
                        // STUB for Phase 4: user-defined function via handle
                        results.Add(UnknownShape)
                    elif Set.contains funcName KNOWN_BUILTINS then
                        // Dispatch to builtin via callback
                        let fakeApplyLine = line
                        let fakeBase = Ir.Var(line, 0, funcName)
                        let builtinResult = builtinDispatch funcName fakeApplyLine fakeBase args env warnings ctx
                        results.Add(builtinResult)
                    elif ctx.ws.externalFunctions.ContainsKey(funcName) then
                        // STUB for Phase 4: external function via handle
                        results.Add(UnknownShape)
                    else
                        results.Add(UnknownShape)
                | _ -> results.Add(UnknownShape)

        // Join all results
        if results.Count = 0 then
            warnings.Value <- warnings.Value @ [ warnLambdaCallApproximate line baseVarName ]
            UnknownShape
        else
            let resultList = results |> Seq.toList
            List.fold joinShape resultList.[0] resultList.[1..]
    | _ -> UnknownShape


/// evalIndexing: indexing logic for Apply-as-indexing nodes.
and private evalIndexing
    (baseShape: Shape)
    (args: IndexArg list)
    (line: int)
    (env: Env)
    (warnings: Diagnostic list ref)
    (ctx: AnalysisContext)
    (builtinDispatch: BuiltinDispatch)
    : Shape =

    match baseShape with
    | UnknownShape -> UnknownShape
    | Scalar -> Scalar   // MATLAB allows indexing scalars: x(1) returns x
    | Matrix(m, n) ->
        match args.Length with
        | 1 ->
            let arg = args.[0]
            match arg with
            | IndexExpr(_, _, (MatrixLit _ as matLitExpr)) ->
                // Matrix literal as linear index: result has the same shape as the index
                evalExprIr matLitExpr env warnings ctx (Some baseShape) builtinDispatch
            | _ -> Scalar

        | 2 ->
            let a1 = args.[0]
            let a2 = args.[1]

            // Bounds checking for IndexExpr arguments
            let mSize = getConcreteDimSize m ctx
            let nSize = getConcreteDimSize n ctx

            match a1, mSize with
            | IndexExpr(_, _, idxExpr), Some ms ->
                let idxIv = getExprInterval idxExpr env ctx
                match idxIv with
                | Some iv ->
                    let fmt = "[" + string iv.lo + ", " + string iv.hi + "]"
                    let dimStr = string ms
                    match iv.lo, iv.hi with
                    | Intervals.Finite lo, _ when lo > ms ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr true ]
                    | _, Intervals.Finite hi when hi < 1 ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr true ]
                    | _, Intervals.Finite hi when hi > ms ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr false ]
                    | Intervals.Finite lo, _ when lo < 1 ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr false ]
                    | _ -> ()
                | None -> ()
            | _ -> ()

            match a2, nSize with
            | IndexExpr(_, _, idxExpr), Some ns ->
                let idxIv = getExprInterval idxExpr env ctx
                match idxIv with
                | Some iv ->
                    let fmt = "[" + string iv.lo + ", " + string iv.hi + "]"
                    let dimStr = string ns
                    match iv.lo, iv.hi with
                    | Intervals.Finite lo, _ when lo > ns ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr true ]
                    | _, Intervals.Finite hi when hi < 1 ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr true ]
                    | _, Intervals.Finite hi when hi > ns ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr false ]
                    | Intervals.Finite lo, _ when lo < 1 ->
                        warnings.Value <- warnings.Value @
                            [ warnIndexOutOfBounds line fmt dimStr false ]
                    | _ -> ()
                | None -> ()
            | _ -> ()

            let evalFn (e: Expr) (en: Env) (w: Diagnostic list ref) (c: AnalysisContext) (cs: Shape option) =
                evalExprIr e en w c cs builtinDispatch

            let rExtent = indexArgToExtentIr a1 env warnings line ctx (Some baseShape) m evalFn
            let cExtent = indexArgToExtentIr a2 env warnings line ctx (Some baseShape) n evalFn

            let isAllowedUnknown (a: IndexArg) =
                match a with
                | Colon _ | Range _ -> true
                | IndexExpr(_, _, MatrixLit _) -> true
                | _ -> false

            let rExtentFinal =
                if rExtent = Unknown && not (isAllowedUnknown a1) then Unknown
                else if isColon a1 then m else rExtent

            let cExtentFinal =
                if cExtent = Unknown && not (isAllowedUnknown a2) then Unknown
                else if isColon a2 then n else cExtent

            if (rExtent = Unknown && not (isAllowedUnknown a1)) ||
               (cExtent = Unknown && not (isAllowedUnknown a2)) then
                UnknownShape
            else
                match rExtentFinal, cExtentFinal with
                | Concrete 1, Concrete 1 -> Scalar
                | _ -> Matrix(rExtentFinal, cExtentFinal)

        | c when c > 2 ->
            warnings.Value <- warnings.Value @
                [ warnTooManyIndices line (Apply(line, 0, Var(line, 0, "?"), args)) ]
            UnknownShape
        | _ -> UnknownShape
    | _ -> UnknownShape


/// isColon: check if an IndexArg is a Colon.
and private isColon (arg: IndexArg) =
    match arg with Colon _ -> true | _ -> false
