module StmtFuncAnalysis

open Ir
open Shapes
open Env
open Context
open Diagnostics
open Builtins
open DimExtract
open Intervals
open SharedTypes
open Constraints
open EvalExpr
open EvalBuiltins

// ---------------------------------------------------------------------------
// StmtFuncAnalysis: statement dispatch + function/lambda call analysis.
// Combined port of analysis/stmt_analysis.py + analysis/func_analysis.py.
// Uses `let rec ... and ...` for mutual recursion.
// ---------------------------------------------------------------------------

// Command-syntax builtins whose OpaqueStmt should not emit W_UNSUPPORTED_STMT.
let private SUPPRESSED_CMD_STMTS : Set<string> =
    Set.ofList [
        "hold"; "grid"; "axis"; "xlabel"; "ylabel"; "zlabel"; "title"; "legend"
        "figure"; "subplot"; "close"; "clf"; "colorbar"; "colormap"; "cla"
        "drawnow"; "pause"; "clear"; "clearvars"; "clc"; "diary"; "dbstop"
        "keyboard"; "set"; "get"; "plot"; "plot3"; "surf"; "mesh"; "contour"
        "imagesc"; "imshow"; "bar"; "histogram"; "scatter"; "stem"; "stairs"
        "disp"; "fprintf"; "warning"; "error"; "print"; "saveas"; "shg"
        "box"; "view"; "lighting"; "material"; "camlight"; "rotate3d"
        "pan"; "zoom"; "format"
        "global"; "persistent"
        "classdef"
        // Workspace/path commands
        "addpath"; "rmpath"; "load"; "save"; "cd"; "pwd"; "mkdir"
        "exist"; "which"; "type"; "doc"; "help"; "lookfor"
        "display"; "input"; "assert"; "narginchk"; "nargoutchk"
    ]


// ---------------------------------------------------------------------------
// Accumulation pattern detection (fixpoint mode only)
// ---------------------------------------------------------------------------

[<Struct>]
type private AccumAxis = Vert | Horz

type private AccumPattern = {
    varName:    string
    axis:       AccumAxis
    deltaExprs: Expr list list  // Non-X rows/elements of MatrixLit
    line:       int
    loopVar:    string
}


/// exprMentionsVar: check if an expression mentions a specific variable.
let rec private exprMentionsVar (expr: Expr) (varName: string) : bool =
    match expr with
    | Var(_, n)        -> n = varName
    | Const _ | StringLit _ | End _ -> false
    | Neg(_, op)       -> exprMentionsVar op varName
    | Not(_, op)       -> exprMentionsVar op varName
    | Transpose(_, op) -> exprMentionsVar op varName
    | BinOp(_, _, l, r) -> exprMentionsVar l varName || exprMentionsVar r varName
    | FieldAccess(_, b, _) -> exprMentionsVar b varName
    | Lambda _ | FuncHandle _ -> false
    | MatrixLit(_, rows) ->
        rows |> List.exists (fun row -> row |> List.exists (fun e -> exprMentionsVar e varName))
    | CellLit(_, rows) ->
        rows |> List.exists (fun row -> row |> List.exists (fun e -> exprMentionsVar e varName))
    | Apply(_, b, iargs) ->
        exprMentionsVar b varName ||
        iargs |> List.exists (fun arg ->
            match arg with IndexExpr(_, e) -> exprMentionsVar e varName | _ -> false)
    | CurlyApply(_, b, iargs) ->
        exprMentionsVar b varName ||
        iargs |> List.exists (fun arg ->
            match arg with IndexExpr(_, e) -> exprMentionsVar e varName | _ -> false)


/// detectAccumulation: detect accumulation patterns in a loop body.
let private detectAccumulation (loopVar: string) (body: Stmt list) : AccumPattern list =
    let candidates = System.Collections.Generic.Dictionary<string, AccumPattern option>()

    for stmt in body do
        match stmt with
        | Assign({ line = stmtLine }, assignName, MatrixLit(_, rows)) ->
            // Count occurrences of Var(name=assignName) in literal
            let mutable count = 0
            for row in rows do
                for elem in row do
                    match elem with
                    | Var(_, n) when n = assignName -> count <- count + 1
                    | _ -> ()

            if count = 1 then
                if candidates.ContainsKey(assignName) then
                    // Multiple assignments: disqualify
                    candidates.[assignName] <- None
                else
                    // Check vertcat or horzcat pattern
                    if rows.Length >= 2 then
                        match rows.[0] with
                        | [ Var(_, n) ] when n = assignName ->
                            let deltaExprs = rows |> List.tail
                            candidates.[assignName] <- Some {
                                varName = assignName; axis = Vert
                                deltaExprs = deltaExprs; line = stmtLine; loopVar = loopVar
                            }
                        | _ -> ()
                    elif rows.Length = 1 && rows.[0].Length >= 2 then
                        match rows.[0].[0] with
                        | Var(_, n) when n = assignName ->
                            let deltaExprs = [ rows.[0] |> List.tail ]
                            candidates.[assignName] <- Some {
                                varName = assignName; axis = Horz
                                deltaExprs = deltaExprs; line = stmtLine; loopVar = loopVar
                            }
                        | _ -> ()
        | _ -> ()

    candidates.Values
    |> Seq.choose id
    |> Seq.toList


/// refineAccumulation: refine accumulation variable shape using algebraic computation.
let private refineAccumulation
    (accum: AccumPattern)
    (iterCount: Dim)
    (preLoopEnv: Env)
    (postLoopEnv: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (wiredEvalExpr: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : unit =
    if iterCount = Unknown then ()
    else
        let initShape = Env.get preLoopEnv accum.varName
        if not (isMatrix initShape) || isUnknown initShape || isBottom initShape then ()
        else
            let currentShape = Env.get postLoopEnv accum.varName
            if not (isMatrix currentShape) then ()
            else
                // Check for self-reference in delta expressions
                let selfRef =
                    accum.deltaExprs |> List.exists (fun row ->
                        row |> List.exists (fun elem ->
                            exprMentionsVar elem accum.varName || exprMentionsVar elem accum.loopVar))
                if selfRef then ()
                else
                    let deltaWarnings = ResizeArray<Diagnostic>()
                    let deltaRows =
                        accum.deltaExprs |> List.map (fun row ->
                            row |> List.map (fun elem -> wiredEvalExpr elem preLoopEnv deltaWarnings ctx))
                    let deltaShape =
                        MatrixLiterals.inferMatrixLiteralShape deltaRows accum.line deltaWarnings ctx preLoopEnv
                    if not (isMatrix deltaShape) then ()
                    else
                        match initShape, deltaShape, currentShape with
                        | Matrix(initRows, initCols), Matrix(deltaRows_, deltaCols), Matrix(curRows, curCols) ->
                            if accum.axis = Vert then
                                if deltaRows_ = Unknown then ()
                                else
                                    let totalAdded = mulDim iterCount deltaRows_
                                    let refinedRows = addDim initRows totalAdded
                                    if curRows = Unknown then
                                        Env.set postLoopEnv accum.varName (Matrix(refinedRows, initCols))
                            else // Horz
                                if deltaCols = Unknown then ()
                                else
                                    let totalAdded = mulDim iterCount deltaCols
                                    let refinedCols = addDim initCols totalAdded
                                    if curCols = Unknown then
                                        Env.set postLoopEnv accum.varName (Matrix(initRows, refinedCols))
                        | _ -> ()


// ---------------------------------------------------------------------------
// Branch join helper
// ---------------------------------------------------------------------------

/// joinBranchResults: join analyzed branches, propagate if all returned.
let private joinBranchResults
    (env: Env)
    (ctx: AnalysisContext)
    (baselineConstraints: System.Collections.Generic.HashSet<string * string>)
    (baselineRanges: System.Collections.Generic.Dictionary<string, SharedTypes.Interval>)
    (branchEnvs: Env list)
    (branchConstraints: System.Collections.Generic.HashSet<string * string> list)
    (branchRanges: System.Collections.Generic.Dictionary<string, SharedTypes.Interval> list)
    (returnedFlags: bool list)
    (deferredExc: System.Exception option)
    : unit =

    if List.forall id returnedFlags then
        match deferredExc with
        | Some exc -> raise exc
        | None -> raise EarlyReturn
    else
        let liveEnvs          = List.zip branchEnvs returnedFlags |> List.choose (fun (e, r) -> if not r then Some e else None)
        let liveConstraints   = List.zip branchConstraints returnedFlags |> List.choose (fun (c, r) -> if not r then Some c else None)
        let liveRanges        = List.zip branchRanges returnedFlags |> List.choose (fun (vr, r) -> if not r then Some vr else None)

        if not liveEnvs.IsEmpty then
            let joined = liveEnvs |> List.fold (fun acc e -> joinEnv acc e) liveEnvs.[0]
            // joinEnv folds from first, so redo properly
            let joinedFinal =
                match liveEnvs with
                | [] -> env
                | first :: rest -> rest |> List.fold joinEnv first
            Env.replaceLocal env joinedFinal

            // Join constraints
            let joinedConstraints = joinConstraints baselineConstraints liveConstraints
            ctx.cst.constraints.Clear()
            for c in joinedConstraints do ctx.cst.constraints.Add(c) |> ignore

            // Provenance pruning
            let newProv = System.Collections.Generic.Dictionary<string * string, int>()
            for c in joinedConstraints do
                match ctx.cst.constraintProvenance.TryGetValue(c) with
                | true, line -> newProv.[c] <- line
                | _ -> ()
            ctx.cst.constraintProvenance.Clear()
            for kv in newProv do ctx.cst.constraintProvenance.[kv.Key] <- kv.Value

            // Join value_ranges
            let joinedRanges = joinValueRanges baselineRanges liveRanges
            ctx.cst.valueRanges.Clear()
            for kv in joinedRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value


// ---------------------------------------------------------------------------
// Dual-location warning formatter
// ---------------------------------------------------------------------------

let private formatDualLocationWarning (funcWarn: Diagnostic) (funcName: string) (callLine: int) : Diagnostic =
    if funcWarn.message.Contains("(in ") then funcWarn
    else
        let augMsg = funcWarn.message + " (in " + funcName + ", called from line " + string callLine + ")"
        { funcWarn with message = augMsg; relatedLine = Some callLine }


// ---------------------------------------------------------------------------
// _update_struct_field helper (mirrors Python's _update_struct_field in dim_extract.py)
// ---------------------------------------------------------------------------

let rec private updateStructField
    (baseShape: Shape)
    (fields: string list)
    (rhsShape: Shape)
    (line: int)
    (warnings: ResizeArray<Diagnostic>)
    : Shape =
    match fields with
    | [] -> rhsShape
    | [fieldName] ->
        match baseShape with
        | Bottom ->
            // Unbound: create new struct
            Struct([ (fieldName, rhsShape) ], false)
        | Struct(existingFields, isOpen) ->
            let fmap = Map.ofList existingFields
            let newMap = Map.add fieldName rhsShape fmap
            Struct(newMap |> Map.toList, isOpen)
        | UnknownShape ->
            // Unknown base: create open struct (has at least this field, may have others)
            Struct([ (fieldName, rhsShape) ], true)
        | _ ->
            // Non-struct: create new struct with just this field
            Struct([ (fieldName, rhsShape) ], false)
    | firstField :: rest ->
        let innerShape =
            match baseShape with
            | Struct(existingFields, _) ->
                let fmap = Map.ofList existingFields
                defaultArg (Map.tryFind firstField fmap) Bottom
            | Bottom -> Bottom
            | _ -> Bottom  // Unknown base: treat inner as Bottom so chain builds correctly
        let updatedInner = updateStructField innerShape rest rhsShape line warnings
        match baseShape with
        | Struct(existingFields, isOpen) ->
            let fmap = Map.ofList existingFields
            let newMap = Map.add firstField updatedInner fmap
            Struct(newMap |> Map.toList, isOpen)
        | Bottom ->
            Struct([ (firstField, updatedInner) ], false)
        | UnknownShape ->
            Struct([ (firstField, updatedInner) ], true)  // open: may have other fields
        | _ ->
            Struct([ (firstField, updatedInner) ], false)


// ---------------------------------------------------------------------------
// Wired eval functions (break circular dependency EvalExpr <-> StmtFuncAnalysis)
// ---------------------------------------------------------------------------

/// wiredEvalExpr: evalExprIr wired with real builtin dispatch and function call stubs.
let rec wiredEvalExpr
    (expr: Expr)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape =
    evalExprIr expr env warnings ctx None wiredBuiltinDispatch

and private wiredBuiltinDispatch
    (fname: string)
    (line: int)
    (baseExpr: Expr)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape =
    ignore baseExpr
    if Set.contains fname KNOWN_BUILTINS then
        evalBuiltinCall fname line args env warnings ctx wiredEvalExprFull wiredGetInterval
    elif ctx.call.functionRegistry.ContainsKey(fname) then
        let sig_ = ctx.call.functionRegistry.[fname]
        if sig_.outputVars.IsEmpty then
            warnings.Add(warnProcedureInExpr line fname)
        let outputShapes = analyzeFunctionCall fname args line env warnings ctx
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
        let sig_ = ctx.call.nestedFunctionRegistry.[fname]
        if sig_.outputVars.IsEmpty then
            warnings.Add(warnProcedureInExpr line fname)
        let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    elif ctx.ws.externalFunctions.ContainsKey(fname) then
        // External functions: no W_PROCEDURE_IN_EXPR (only applies to same-file functions)
        let extSig = ctx.ws.externalFunctions.[fname]
        let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    else
        warnings.Add(warnUnknownFunction line fname)
        UnknownShape

and private wiredGetInterval
    (expr: Expr)
    (env: Env)
    (ctx: AnalysisContext)
    : Interval option =
    getExprInterval expr env ctx

// Override evalApply to wire function call stubs to real implementations.
// This is done via the builtinDispatch callback mechanism.
// For user-defined function calls, we also need to route through wiredCallUserFunc.
// Unfortunately EvalExpr's evalApply has stubs for user/external calls.
// We patch by re-implementing wiredEvalExpr to redirect Apply nodes to our handlers.
and private wiredEvalExprFull
    (expr: Expr)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape =
    match expr with
    | Apply({ line = line }, Var(_, fname), args) ->
        // Determine dispatch priority
        let varShape = Env.get env fname
        if isFunctionHandle varShape then
            // Priority 1: function handle variable
            evalExprIr expr env warnings ctx None wiredBuiltinDispatch
        elif Set.contains fname KNOWN_BUILTINS then
            // Priority 2: builtin
            wiredBuiltinDispatch fname line (Var(loc line 0, fname)) args env warnings ctx
        elif not (Env.hasLocal env fname) then
            // Priority 3-5: unbound — user function, nested, external, or unknown
            if ctx.call.functionRegistry.ContainsKey(fname) then
                let sig_ = ctx.call.functionRegistry.[fname]
                if sig_.outputVars.IsEmpty then
                    warnings.Add(warnProcedureInExpr line fname)
                let outputShapes = analyzeFunctionCall fname args line env warnings ctx
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
                let sig_ = ctx.call.nestedFunctionRegistry.[fname]
                if sig_.outputVars.IsEmpty then
                    warnings.Add(warnProcedureInExpr line fname)
                let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            elif ctx.ws.externalFunctions.ContainsKey(fname) then
                // External functions: no W_PROCEDURE_IN_EXPR (only applies to same-file functions)
                let extSig = ctx.ws.externalFunctions.[fname]
                let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            else
                warnings.Add(warnUnknownFunction line fname)
                UnknownShape
        else
            // Priority 6: bound non-handle variable — treat as indexing
            evalExprIr expr env warnings ctx None wiredBuiltinDispatch
    | _ ->
        evalExprIr expr env warnings ctx None wiredBuiltinDispatch


// ---------------------------------------------------------------------------
// analyzeStmtIr: main statement dispatcher
// ---------------------------------------------------------------------------

and analyzeStmtIr
    (stmt: Stmt)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : unit =

    match stmt with
    | Assign({ line = line }, name, expr) ->
        let oldShape = Env.get env name
        let newShape = wiredEvalExprFull expr env warnings ctx

        if Env.hasLocal env name && AnalysisCore.shapesDefinitelyIncompatible oldShape newShape then
            warnings.Add(warnReassignIncompatible line name newShape oldShape)

        Env.set env name newShape

        // Constraint validation: first binding of scalar to concrete value
        if isBottom oldShape && isScalar newShape then
            let concreteValue = tryExtractConstValue expr
            match concreteValue with
            | Some cv ->
                ctx.cst.scalarBindings.[name] <- cv
                validateBinding ctx env name cv warnings line
            | None -> ()

        // Interval tracking
        if isScalar newShape then
            let iv = getExprInterval expr env ctx
            match iv with
            | Some i -> ctx.cst.valueRanges.[name] <- i
            | None ->
                if ctx.cst.valueRanges.ContainsKey(name) then
                    ctx.cst.valueRanges.Remove(name) |> ignore
        else
            if ctx.cst.valueRanges.ContainsKey(name) then
                ctx.cst.valueRanges.Remove(name) |> ignore

    | StructAssign({ line = line }, baseName, fields, expr) ->
        let rhsShape = wiredEvalExprFull expr env warnings ctx
        let baseShape = Env.get env baseName
        let updated = updateStructField baseShape fields rhsShape line warnings
        Env.set env baseName updated

    | FieldIndexAssign({ line = line }, baseName, prefixFields, indexArgs, _, suffixFields, expr) ->
        let rhsShape = wiredEvalExprFull expr env warnings ctx
        let baseShape = Env.get env baseName
        // Evaluate index args for side effects
        for arg in indexArgs do
            evalArgShapeHelper arg env warnings ctx baseShape |> ignore
        let allFields = prefixFields @ suffixFields
        let updated = updateStructField baseShape allFields rhsShape line warnings
        Env.set env baseName updated

    | CellAssign({ line = line }, baseName, args, expr) ->
        let rhsShape = wiredEvalExprFull expr env warnings ctx
        let baseShape = Env.get env baseName
        if isBottom baseShape then
            Env.set env baseName (Cell(Unknown, Unknown, None))
        elif not (isCell baseShape) then
            if isEmptyMatrix baseShape then
                Env.set env baseName (Cell(Unknown, Unknown, None))
                // Fall through: now it's a cell
                analyzeCellAssignArgs baseName args env warnings ctx rhsShape
            else
                for arg in args do evalArgShapeHelper arg env warnings ctx baseShape |> ignore
                if not (isUnknown baseShape) then
                    warnings.Add(warnCellAssignNonCell line baseName baseShape)
        else
            analyzeCellAssignArgs baseName args env warnings ctx rhsShape

    | IndexAssign(_, baseName, _, expr) ->
        let rhsShape = wiredEvalExprFull expr env warnings ctx
        ignore rhsShape
        let baseShape = Env.get env baseName
        if isBottom baseShape then
            Env.set env baseName UnknownShape
        else
            let isIndexable =
                isMatrix baseShape || isUnknown baseShape || isScalar baseShape ||
                isStruct baseShape || isEmptyMatrix baseShape
            if not isIndexable then
                warnings.Add(warnIndexAssignTypeMismatch (stmt.Line) baseName baseShape)
        // No OOB checking: MATLAB auto-expands on indexed assign

    | IndexStructAssign(_, baseName, _, _, _, expr) ->
        wiredEvalExprFull expr env warnings ctx |> ignore
        let existing = Env.get env baseName
        if isBottom existing then Env.set env baseName UnknownShape

    | ExprStmt(_, expr) ->
        wiredEvalExprFull expr env warnings ctx |> ignore

    | While({ line = line }, cond, body) ->
        wiredEvalExprFull cond env warnings ctx |> ignore

        let baselineRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)
        let refinements = extractConditionRefinements cond env ctx
        applyRefinements ctx refinements false

        analyzeLoopBody body env warnings ctx None

        ctx.cst.valueRanges.Clear()
        for kv in baselineRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value

    | For(_, var_, it, body) ->
        Env.set env var_ Scalar
        wiredEvalExprFull it env warnings ctx |> ignore

        // Record loop variable interval
        match it with
        | BinOp(_, ":", left, right) ->
            let loConst = tryExtractConstValue left
            let hiConst = tryExtractConstValue right
            match loConst, hiConst with
            | Some lo, Some hi ->
                ctx.cst.valueRanges.[var_] <- { lo = Finite lo; hi = Finite hi }
            | _ ->
                let loDim = exprToDimIr left env
                let hiDim = exprToDimIr right env
                let loBound =
                    match loConst with
                    | Some v -> Finite v
                    | None -> match loDim with Concrete n -> Finite n | Symbolic s -> SymBound s | Unknown -> Unbounded
                let hiBound =
                    match hiConst with
                    | Some v -> Finite v
                    | None -> match hiDim with Concrete n -> Finite n | Symbolic s -> SymBound s | Unknown -> Unbounded
                ctx.cst.valueRanges.[var_] <- { lo = loBound; hi = hiBound }
        | _ -> ()

        // Fixpoint-only: accumulation refinement
        let (preLoopEnv, iterCount, accumPatterns) =
            if ctx.call.fixpoint then
                let pre = Env.copy env
                let ic = extractIterationCount it env (Some ctx)
                let accum = if ic <> Unknown then detectAccumulation var_ body else []
                (pre, ic, accum)
            else
                (Env.copy env, Unknown, [])

        analyzeLoopBody body env warnings ctx (Some var_)

        if ctx.call.fixpoint then
            for accum in accumPatterns do
                refineAccumulation accum iterCount preLoopEnv env warnings ctx wiredEvalExprFull

    | If({ line = line }, cond, thenBody, elseBody) ->
        wiredEvalExprFull cond env warnings ctx |> ignore

        let refinements = extractConditionRefinements cond env ctx
        let baselineConstraints = snapshotConstraints ctx
        let baselineRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        let thenEnv = Env.copy env
        let elseEnv = Env.copy env

        let mutable thenReturned = false
        let mutable elseReturned = false

        // Then branch
        applyRefinements ctx refinements false
        ctx.cst.pathConstraints.Push(cond, true, line)
        try
            for s in thenBody do analyzeStmtIr s thenEnv warnings ctx
        with
        | :? EarlyReturn -> thenReturned <- true
        | :? EarlyBreak as e -> thenReturned <- true; ctx.cst.pathConstraints.Pop(); reraise ()
        | :? EarlyContinue as e -> thenReturned <- true; ctx.cst.pathConstraints.Pop(); reraise ()
        ctx.cst.pathConstraints.Pop()
        let thenConstraints = snapshotConstraints ctx
        let thenRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        // Reset to baseline for else branch
        ctx.cst.constraints.Clear()
        for c in baselineConstraints do ctx.cst.constraints.Add(c) |> ignore
        ctx.cst.valueRanges.Clear()
        for kv in baselineRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value

        // Else branch
        applyRefinements ctx refinements true
        ctx.cst.pathConstraints.Push(cond, false, line)
        try
            for s in elseBody do analyzeStmtIr s elseEnv warnings ctx
        with
        | :? EarlyReturn -> elseReturned <- true
        | :? EarlyBreak as e -> elseReturned <- true; ctx.cst.pathConstraints.Pop(); reraise ()
        | :? EarlyContinue as e -> elseReturned <- true; ctx.cst.pathConstraints.Pop(); reraise ()
        ctx.cst.pathConstraints.Pop()
        let elseConstraints = snapshotConstraints ctx
        let elseRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        joinBranchResults
            env ctx baselineConstraints baselineRanges
            [thenEnv; elseEnv] [thenConstraints; elseConstraints]
            [thenRanges; elseRanges] [thenReturned; elseReturned] None

    | IfChain({ line = line }, conditions, bodies, elseBody) ->
        for cond in conditions do wiredEvalExprFull cond env warnings ctx |> ignore

        let allRefinements = conditions |> List.map (fun cond -> extractConditionRefinements cond env ctx)
        let baselineConstraints = snapshotConstraints ctx
        let baselineRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        let allBodies = bodies @ [ elseBody ]
        let mutable branchEnvs : Env list = []
        let mutable branchConstraints : System.Collections.Generic.HashSet<string * string> list = []
        let mutable branchRanges : System.Collections.Generic.Dictionary<string, SharedTypes.Interval> list = []
        let mutable returnedFlags : bool list = []
        let mutable deferredExc : System.Exception option = None

        for idx in 0 .. allBodies.Length - 1 do
            ctx.cst.constraints.Clear()
            for c in baselineConstraints do ctx.cst.constraints.Add(c) |> ignore
            ctx.cst.valueRanges.Clear()
            for kv in baselineRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value

            if idx < allRefinements.Length then
                applyRefinements ctx allRefinements.[idx] false
                ctx.cst.pathConstraints.Push(conditions.[idx], true, line)

            let branchEnv = Env.copy env
            let mutable returned = false
            try
                for s in allBodies.[idx] do analyzeStmtIr s branchEnv warnings ctx
            with
            | :? EarlyReturn -> returned <- true
            | :? EarlyBreak as e ->
                returned <- true
                deferredExc <- Some (e :> System.Exception)
            | :? EarlyContinue as e ->
                returned <- true
                deferredExc <- Some (e :> System.Exception)
            if idx < conditions.Length then ctx.cst.pathConstraints.Pop()

            branchEnvs <- branchEnvs @ [branchEnv]
            branchConstraints <- branchConstraints @ [ snapshotConstraints ctx ]
            branchRanges <- branchRanges @ [ System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges) ]
            returnedFlags <- returnedFlags @ [returned]

        joinBranchResults
            env ctx baselineConstraints baselineRanges
            branchEnvs branchConstraints branchRanges returnedFlags deferredExc

    | Switch({ line = line }, expr, cases, otherwise) ->
        wiredEvalExprFull expr env warnings ctx |> ignore
        for (caseVal, _) in cases do wiredEvalExprFull caseVal env warnings ctx |> ignore

        // Extract the switch variable name for interval refinement (simple Var only).
        let switchVarName =
            match expr with
            | Ir.Var(_, name) -> Some name
            | _ -> None

        let baselineConstraints = snapshotConstraints ctx
        let baselineRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        let allBodies = (cases |> List.map snd) @ [ otherwise ]
        let mutable branchEnvs : Env list = []
        let mutable branchConstraints : System.Collections.Generic.HashSet<string * string> list = []
        let mutable branchRanges : System.Collections.Generic.Dictionary<string, SharedTypes.Interval> list = []
        let mutable returnedFlags : bool list = []
        let mutable deferredExc : System.Exception option = None

        for caseIdx in 0 .. allBodies.Length - 1 do
            ctx.cst.constraints.Clear()
            for c in baselineConstraints do ctx.cst.constraints.Add(c) |> ignore
            ctx.cst.valueRanges.Clear()
            for kv in baselineRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value

            let isCase = caseIdx < cases.Length
            if isCase then
                let (caseCond, _) = cases.[caseIdx]
                ctx.cst.pathConstraints.Push(caseCond, true, line)
                // Narrow the switch variable's interval when case value is a constant.
                match switchVarName, caseCond with
                | Some varName, Ir.Const(_, v)
                    when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) ->
                    let refinements = [ (varName, "==", Shapes.Concrete (int v)) ]
                    Intervals.applyRefinements ctx refinements false
                | _ -> ()

            let branchEnv = Env.copy env
            let mutable returned = false
            try
                for s in allBodies.[caseIdx] do analyzeStmtIr s branchEnv warnings ctx
            with
            | :? EarlyReturn -> returned <- true
            | :? EarlyBreak as e ->
                returned <- true
                deferredExc <- Some (e :> System.Exception)
            | :? EarlyContinue as e ->
                returned <- true
                deferredExc <- Some (e :> System.Exception)
            if isCase then ctx.cst.pathConstraints.Pop()

            branchEnvs <- branchEnvs @ [branchEnv]
            branchConstraints <- branchConstraints @ [ snapshotConstraints ctx ]
            branchRanges <- branchRanges @ [ System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges) ]
            returnedFlags <- returnedFlags @ [returned]

        joinBranchResults
            env ctx baselineConstraints baselineRanges
            branchEnvs branchConstraints branchRanges returnedFlags deferredExc

    | Try(_, tryBody, catchBody) ->
        let baselineConstraints = snapshotConstraints ctx
        let baselineRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)
        let preTryEnv = Env.copy env

        let tryEnv = Env.copy env
        let mutable tryReturned = false
        let mutable deferredExc : System.Exception option = None
        try
            for s in tryBody do analyzeStmtIr s tryEnv warnings ctx
        with
        | :? EarlyReturn -> tryReturned <- true
        | :? EarlyBreak as e -> tryReturned <- true; deferredExc <- Some (e :> System.Exception)
        | :? EarlyContinue as e -> tryReturned <- true; deferredExc <- Some (e :> System.Exception)
        let tryConstraints = snapshotConstraints ctx
        let tryRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        ctx.cst.constraints.Clear()
        for c in baselineConstraints do ctx.cst.constraints.Add(c) |> ignore
        ctx.cst.valueRanges.Clear()
        for kv in baselineRanges do ctx.cst.valueRanges.[kv.Key] <- kv.Value

        let catchEnv = Env.copy preTryEnv
        let mutable catchReturned = false
        try
            for s in catchBody do analyzeStmtIr s catchEnv warnings ctx
        with
        | :? EarlyReturn -> catchReturned <- true
        | :? EarlyBreak as e ->
            catchReturned <- true
            if deferredExc.IsNone then deferredExc <- Some (e :> System.Exception)
        | :? EarlyContinue as e ->
            catchReturned <- true
            if deferredExc.IsNone then deferredExc <- Some (e :> System.Exception)
        let catchConstraints = snapshotConstraints ctx
        let catchRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        joinBranchResults
            env ctx baselineConstraints baselineRanges
            [tryEnv; catchEnv] [tryConstraints; catchConstraints]
            [tryRanges; catchRanges] [tryReturned; catchReturned] deferredExc

    | OpaqueStmt({ line = line }, targets, raw) ->
        let firstWord = if raw.Trim() = "" then "" else raw.Trim().Split([| ' '; '\t' |]).[0]
        if not (Set.contains firstWord SUPPRESSED_CMD_STMTS) then
            warnings.Add(warnUnsupportedStmt line raw targets)
        for targetName in targets do Env.set env targetName UnknownShape

    | Return _ -> raise EarlyReturn
    | Break _  -> raise EarlyBreak
    | Continue _ -> raise EarlyContinue

    | FunctionDef _ ->
        // No-op: function defs are pre-scanned in pass 1
        ()

    | AssignMulti({ line = line }, targets, expr) ->
        analyzeAssignMulti line targets expr env warnings ctx


// ---------------------------------------------------------------------------
// CellAssign helper
// ---------------------------------------------------------------------------

and private evalArgShapeHelper
    (arg: IndexArg)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (containerShape: Shape)
    : Shape =
    evalIndexArgToShape arg env warnings ctx (Some containerShape)
        (fun e en w c cs -> evalExprIr e en w c cs wiredBuiltinDispatch)

and private analyzeCellAssignArgs
    (baseName: string)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (rhsShape: Shape)
    : unit =
    let baseShape = Env.get env baseName
    if not (isCell baseShape) then ()
    else
        match baseShape with
        | Cell(rows, cols, elemMap) ->
            if args.Length = 1 then
                match args.[0] with
                | IndexExpr(_, Const(_, v)) ->
                    let idx0 = int v - 1
                    let currentElems = defaultArg elemMap Map.empty
                    let newElems = Map.add idx0 rhsShape currentElems
                    Env.set env baseName (Cell(rows, cols, Some newElems))
                | _ ->
                    for arg in args do evalArgShapeHelper arg env warnings ctx baseShape |> ignore
                    Env.set env baseName (Cell(rows, cols, None))
            else
                for arg in args do evalArgShapeHelper arg env warnings ctx baseShape |> ignore
                Env.set env baseName (Cell(rows, cols, None))
        | _ -> ()


// ---------------------------------------------------------------------------
// AssignMulti dispatcher
// ---------------------------------------------------------------------------

and private analyzeAssignMulti
    (line: int)
    (targets: string list)
    (expr: Expr)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : unit =

    let bindTarget (target: string) (shape: Shape) =
        if target = "~" then ()
        elif target.Contains(".") then
            let parts = target.Split('.') |> Array.toList
            let baseName = parts.[0]
            let fields = parts.[1..]
            let baseShape = Env.get env baseName
            let updated = updateStructField baseShape fields shape line warnings
            Env.set env baseName updated
        else
            Env.set env target shape

    match expr with
    | Apply(_, Var(_, fname), args) ->
        if Set.contains fname KNOWN_BUILTINS then
            let numTargets = targets.Length
            if numTargets = 1 then
                let result = evalBuiltinCall fname line args env warnings ctx wiredEvalExprFull wiredGetInterval
                bindTarget targets.[0] result
            else
                match evalMultiBuiltinCall fname numTargets args env warnings ctx wiredEvalExprFull wiredGetInterval with
                | Some shapes ->
                    for (target, shape) in List.zip targets shapes do bindTarget target shape
                | None ->
                    let supported = defaultArg (Map.tryFind fname MULTI_SUPPORTED_FORMS) "unknown"
                    warnings.Add(warnMultiReturnCount line fname supported numTargets)
                    for target in targets do bindTarget target UnknownShape

        elif ctx.call.functionRegistry.ContainsKey(fname) then
            let outputShapes = analyzeFunctionCall fname args line env warnings ctx
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
            let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        elif ctx.ws.externalFunctions.ContainsKey(fname) then
            let extSig = ctx.ws.externalFunctions.[fname]
            let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        else
            warnings.Add(warnUnknownFunction line fname)
            for target in targets do bindTarget target UnknownShape

    | _ ->
        warnings.Add(warnMultiAssignNonCall line)
        for target in targets do bindTarget target UnknownShape


// ---------------------------------------------------------------------------
// analyzeLoopBody: 3-phase widening algorithm (fixpoint mode)
// ---------------------------------------------------------------------------

and analyzeLoopBody
    (body: Stmt list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (loopVar: string option)
    : unit =

    if not ctx.call.fixpoint then
        try
            for s in body do analyzeStmtIr s env warnings ctx
        with
        | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()
    else
        // Snapshot value ranges before loop body (mirrors preLoopEnv for shapes)
        let preLoopRanges = System.Collections.Generic.Dictionary<string, SharedTypes.Interval>(ctx.cst.valueRanges)

        // Phase 1 (Discover)
        let preLoopEnv = Env.copy env
        try
            for s in body do analyzeStmtIr s env warnings ctx
        with
        | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()

        // Widen shapes
        let widened = widenEnv preLoopEnv env
        // Widen intervals in parallel with shape widening
        widenValueRanges preLoopRanges ctx.cst.valueRanges loopVar

        // Phase 2 (Stabilize): Re-analyze if widening changed anything
        if not (Env.localBindingsEqual env widened) then
            Env.replaceLocal env widened
            try
                for s in body do analyzeStmtIr s env warnings ctx
            with
            | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()

        // Widen intervals again before post-loop join (mirrors finalWidened for shapes)
        widenValueRanges preLoopRanges ctx.cst.valueRanges loopVar

        // Phase 3 (Post-loop join): Model "loop may execute 0 times"
        let finalWidened = widenEnv preLoopEnv env
        Env.replaceLocal env finalWidened


// ---------------------------------------------------------------------------
// analyzeFunctionCall: interprocedural analysis with polymorphic caching
// ---------------------------------------------------------------------------

and analyzeFunctionCall
    (funcName: string)
    (args: IndexArg list)
    (line: int)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape list =

    if not (ctx.call.functionRegistry.ContainsKey(funcName)) then [ UnknownShape ]
    else
        let sig_ = ctx.call.functionRegistry.[funcName]

        // Check argument count
        if args.Length <> sig_.parms.Length then
            warnings.Add(warnFunctionArgCountMismatch line funcName sig_.parms.Length args.Length)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        // Recursion guard
        if ctx.call.analyzingFunctions.Contains(funcName) then
            warnings.Add(warnRecursiveFunction line funcName)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        // Evaluate arg shapes for cache key
        let argShapes =
            args |> List.map (fun arg ->
                match arg with
                | IndexExpr(_, e) -> wiredEvalExprFull e env warnings ctx
                | _ -> UnknownShape)

        let argDimAliases =
            List.map2 (fun (param: string) arg ->
                match arg with
                | IndexExpr(_, e) -> (param, exprToDimIr e env)
                | _ -> (param, Unknown)) sig_.parms args

        let cacheKey =
            funcName + ":(" +
            (argShapes |> List.map shapeToString |> String.concat ",") + "):(" +
            (argDimAliases |> List.map (fun (p, d) -> p + "=" + dimStr d) |> String.concat ",") + ")"

        match ctx.call.analysisCache.TryGetValue(cacheKey) with
        | true, (:? (Shape list * Diagnostic list) as cached) ->
            let (cachedShapes, cachedWarn) = cached
            for fw in cachedWarn do
                warnings.Add(formatDualLocationWarning fw funcName line)
            List.ofSeq cachedShapes
        | _ ->
            ctx.call.analyzingFunctions.Add(funcName) |> ignore
            try
                ctx.SnapshotScope(fun () ->
                    let funcEnv = Env.create ()
                    let funcWarnings = ResizeArray<Diagnostic>()

                    // Bind parameters
                    for (param, arg, argShape) in List.zip3 sig_.parms args argShapes do
                        Env.set funcEnv param argShape
                        match arg with
                        | IndexExpr(_, e) ->
                            let callerDim = exprToDimIr e env
                            if callerDim <> Unknown then
                                funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                        | _ -> ()

                    // Pre-scan body for nested FunctionDefs
                    for s in sig_.body do
                        match s with
                        | FunctionDef({ line = nLine; col = nCol }, nestedName, nestedParms, nestedOuts, nestedBody) ->
                            ctx.call.nestedFunctionRegistry.[nestedName] <-
                                { name = nestedName; parms = nestedParms; outputVars = nestedOuts; body = nestedBody; defLine = nLine; defCol = nCol }
                        | _ -> ()

                    // Analyze function body
                    try
                        for s in sig_.body do analyzeStmtIr s funcEnv funcWarnings ctx
                    with :? EarlyReturn -> ()

                    // Extract return values
                    let resultShapes =
                        sig_.outputVars
                        |> List.map (fun outVar ->
                            let shape = Env.get funcEnv outVar
                            if isBottom shape then UnknownShape else shape)
                    let result = if resultShapes.IsEmpty then [ UnknownShape ] else resultShapes

                    ctx.call.analysisCache.[cacheKey] <- box (result, Seq.toList funcWarnings)

                    for fw in funcWarnings do
                        warnings.Add(formatDualLocationWarning fw funcName line)

                    result)
            finally
                ctx.call.analyzingFunctions.Remove(funcName) |> ignore


// ---------------------------------------------------------------------------
// analyzeNestedFunctionCall
// ---------------------------------------------------------------------------

and analyzeNestedFunctionCall
    (funcName: string)
    (args: IndexArg list)
    (line: int)
    (parentEnv: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape list =

    if not (ctx.call.nestedFunctionRegistry.ContainsKey(funcName)) then [ UnknownShape ]
    else
        let sig_ = ctx.call.nestedFunctionRegistry.[funcName]

        if args.Length <> sig_.parms.Length then
            warnings.Add(warnFunctionArgCountMismatch line funcName sig_.parms.Length args.Length)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        if ctx.call.analyzingFunctions.Contains(funcName) then
            warnings.Add(warnRecursiveFunction line funcName)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        let argShapes =
            args |> List.map (fun arg ->
                match arg with
                | IndexExpr(_, e) -> wiredEvalExprFull e parentEnv warnings ctx
                | _ -> UnknownShape)

        let argDimAliases =
            List.map2 (fun (param: string) arg ->
                match arg with
                | IndexExpr(_, e) -> (param, exprToDimIr e parentEnv)
                | _ -> (param, Unknown)) sig_.parms args

        let cacheKey =
            "nested:" + funcName + ":(" +
            (argShapes |> List.map shapeToString |> String.concat ",") + "):(" +
            (argDimAliases |> List.map (fun (p, d) -> p + "=" + dimStr d) |> String.concat ",") + ")"

        match ctx.call.analysisCache.TryGetValue(cacheKey) with
        | true, (:? (Shape list * Diagnostic list) as cached) ->
            let (cachedShapes, cachedWarn) = cached
            for fw in cachedWarn do
                warnings.Add(formatDualLocationWarning fw funcName line)
            List.ofSeq cachedShapes
        | _ ->
            ctx.call.analyzingFunctions.Add(funcName) |> ignore
            try
                ctx.SnapshotScope(fun () ->
                    let funcEnv = Env.createWithParent parentEnv
                    let funcWarnings = ResizeArray<Diagnostic>()

                    for (param, arg, argShape) in List.zip3 sig_.parms args argShapes do
                        Env.set funcEnv param argShape
                        match arg with
                        | IndexExpr(_, e) ->
                            let callerDim = exprToDimIr e parentEnv
                            if callerDim <> Unknown then
                                funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                        | _ -> ()

                    for s in sig_.body do
                        match s with
                        | FunctionDef({ line = nLine; col = nCol }, nestedName, nestedParms, nestedOuts, nestedBody) ->
                            ctx.call.nestedFunctionRegistry.[nestedName] <-
                                { name = nestedName; parms = nestedParms; outputVars = nestedOuts; body = nestedBody; defLine = nLine; defCol = nCol }
                        | _ -> ()

                    try
                        for s in sig_.body do analyzeStmtIr s funcEnv funcWarnings ctx
                    with :? EarlyReturn -> ()

                    // Write-back: flush modified parent-visible variables
                    let paramSet = Set.ofList sig_.parms
                    for kv in funcEnv.bindings do
                        if not (Set.contains kv.Key paramSet) && Env.contains parentEnv kv.Key then
                            Env.set parentEnv kv.Key kv.Value

                    let resultShapes =
                        sig_.outputVars
                        |> List.map (fun outVar ->
                            let shape = Env.get funcEnv outVar
                            if isBottom shape then UnknownShape else shape)
                    let result = if resultShapes.IsEmpty then [ UnknownShape ] else resultShapes

                    ctx.call.analysisCache.[cacheKey] <- box (result, Seq.toList funcWarnings)

                    for fw in funcWarnings do
                        warnings.Add(formatDualLocationWarning fw funcName line)

                    result)
            finally
                ctx.call.analyzingFunctions.Remove(funcName) |> ignore


// ---------------------------------------------------------------------------
// analyzeExternalFunctionCall: cross-file analysis
// ---------------------------------------------------------------------------

and analyzeExternalFunctionCall
    (fname: string)
    (extSig: ExternalSignature)
    (args: IndexArg list)
    (line: int)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    : Shape list =

    // Cross-file recursion guard
    if ctx.ws.analyzingExternal.Contains(fname) then
        List.replicate (max extSig.returnCount 1) UnknownShape
    else

    // Load and parse external file
    let loaded = Workspace.loadExternalFunction extSig
    match loaded with
    | None ->
        warnings.Add(warnExternalParseError line fname extSig.filename)
        List.replicate (max extSig.returnCount 1) UnknownShape
    | Some (primarySig, (subfunctions : Map<string, FunctionSignature>)) ->

    // Arg count check
    if args.Length <> primarySig.parms.Length then
        warnings.Add(warnFunctionArgCountMismatch line fname primarySig.parms.Length args.Length)
        List.replicate (max primarySig.outputVars.Length 1) UnknownShape
    else

    let argShapes =
        args |> List.map (fun arg ->
            match arg with
            | IndexExpr(_, e) -> wiredEvalExprFull e env warnings ctx
            | _ -> UnknownShape)

    let argDimAliases =
        List.map2 (fun (param: string) arg ->
            match arg with
            | IndexExpr(_, e) -> (param, exprToDimIr e env)
            | _ -> (param, Unknown)) primarySig.parms args

    let cacheKey =
        "external:" + fname + ":(" +
        (argShapes |> List.map shapeToString |> String.concat ",") + "):(" +
        (argDimAliases |> List.map (fun (p, d) -> p + "=" + dimStr d) |> String.concat ",") + ")"

    match ctx.call.analysisCache.TryGetValue(cacheKey) with
    | true, (:? (Shape list * Diagnostic list) as cached) ->
        let (cachedShapes, _) = cached
        List.ofSeq cachedShapes  // External warnings suppressed
    | _ ->
        // Registry swap + recursion guard
        let savedRegistry = ctx.call.functionRegistry
        ctx.call.functionRegistry <- System.Collections.Generic.Dictionary<string, FunctionSignature>()
        for kv in subfunctions do ctx.call.functionRegistry.[kv.Key] <- kv.Value
        ctx.ws.analyzingExternal.Add(fname) |> ignore
        ctx.call.analyzingFunctions.Add(fname) |> ignore

        try
            ctx.SnapshotScope(fun () ->
                let funcEnv = Env.create ()
                let funcWarnings = ResizeArray<Diagnostic>()  // Suppressed

                for (param, arg, argShape) in List.zip3 primarySig.parms args argShapes do
                    Env.set funcEnv param argShape
                    match arg with
                    | IndexExpr(_, e) ->
                        let callerDim = exprToDimIr e env
                        if callerDim <> Unknown then
                            funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                    | _ -> ()

                try
                    for s in primarySig.body do analyzeStmtIr s funcEnv funcWarnings ctx
                with :? EarlyReturn -> ()

                let resultShapes =
                    primarySig.outputVars
                    |> List.map (fun outVar ->
                        let shape = Env.get funcEnv outVar
                        if isBottom shape then UnknownShape else shape)
                let result = if resultShapes.IsEmpty then [ UnknownShape ] else resultShapes

                ctx.call.analysisCache.[cacheKey] <- box (result, [])
                result)
        finally
            ctx.call.functionRegistry <- savedRegistry
            ctx.ws.analyzingExternal.Remove(fname) |> ignore
            ctx.call.analyzingFunctions.Remove(fname) |> ignore
