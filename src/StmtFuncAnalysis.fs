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


/// exprMentionsFieldAccess: check if an expression mentions a specific struct field access.
let rec private exprMentionsFieldAccess (expr: Expr) (baseName: string) (fieldName: string) : bool =
    match expr with
    | FieldAccess(_, Var(_, bn), fn) -> bn = baseName && fn = fieldName
    | Var _ | Const _ | StringLit _ | End _ -> false
    | Neg(_, op)       -> exprMentionsFieldAccess op baseName fieldName
    | Not(_, op)       -> exprMentionsFieldAccess op baseName fieldName
    | Transpose(_, op) -> exprMentionsFieldAccess op baseName fieldName
    | BinOp(_, _, l, r) ->
        exprMentionsFieldAccess l baseName fieldName ||
        exprMentionsFieldAccess r baseName fieldName
    | FieldAccess(_, b, _) -> exprMentionsFieldAccess b baseName fieldName
    | Lambda _ | FuncHandle _ -> false
    | MatrixLit(_, rows) ->
        rows |> List.exists (fun row -> row |> List.exists (fun e -> exprMentionsFieldAccess e baseName fieldName))
    | CellLit(_, rows) ->
        rows |> List.exists (fun row -> row |> List.exists (fun e -> exprMentionsFieldAccess e baseName fieldName))
    | Apply(_, bexpr, iargs) ->
        exprMentionsFieldAccess bexpr baseName fieldName ||
        iargs |> List.exists (fun arg ->
            match arg with IndexExpr(_, e) -> exprMentionsFieldAccess e baseName fieldName | _ -> false)
    | CurlyApply(_, bexpr, iargs) ->
        exprMentionsFieldAccess bexpr baseName fieldName ||
        iargs |> List.exists (fun arg ->
            match arg with IndexExpr(_, e) -> exprMentionsFieldAccess e baseName fieldName | _ -> false)


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

        | StructAssign({ line = stmtLine }, baseName, [fieldName], MatrixLit(_, rows)) ->
            // Struct field accumulation: s.field = [s.field; row] or s.field = [s.field, col]
            let key = baseName + "." + fieldName
            let mutable count = 0
            for row in rows do
                for elem in row do
                    if exprMentionsFieldAccess elem baseName fieldName then
                        count <- count + 1

            if count = 1 then
                if candidates.ContainsKey(key) then
                    candidates.[key] <- None
                else
                    if rows.Length >= 2 then
                        match rows.[0] with
                        | [ singleElem ] when exprMentionsFieldAccess singleElem baseName fieldName ->
                            let deltaExprs = rows |> List.tail
                            candidates.[key] <- Some {
                                varName = key; axis = Vert
                                deltaExprs = deltaExprs; line = stmtLine; loopVar = loopVar
                            }
                        | _ -> ()
                    elif rows.Length = 1 && rows.[0].Length >= 2 then
                        if exprMentionsFieldAccess rows.[0].[0] baseName fieldName then
                            let deltaExprs = [ rows.[0] |> List.tail ]
                            candidates.[key] <- Some {
                                varName = key; axis = Horz
                                deltaExprs = deltaExprs; line = stmtLine; loopVar = loopVar
                            }

        | _ -> ()

    candidates.Values
    |> Seq.choose id
    |> Seq.toList


/// getStructField: extract the shape of a named field from a Struct shape. Returns Bottom if missing.
let private getStructField (structShape: Shape) (fieldName: string) : Shape =
    match structShape with
    | Struct(fields, _) ->
        fields |> List.tryFind (fun (fn, _) -> fn = fieldName) |> Option.map snd |> Option.defaultValue Bottom
    | _ -> Bottom

/// setStructField: return a new Struct shape with a specific field updated.
let private setStructField (structShape: Shape) (fieldName: string) (fieldShape: Shape) : Shape =
    match structShape with
    | Struct(fields, isOpen) ->
        let fmap = Map.ofList fields
        let newMap = Map.add fieldName fieldShape fmap
        Struct(newMap |> Map.toList, isOpen)
    | _ -> structShape  // Can't update field on non-struct; return unchanged


/// refineAccumulation: refine accumulation variable shape using algebraic computation.
/// Handles plain variables and dot-separated struct fields (baseName.fieldName).
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
        // Determine whether this is a struct field accumulation (varName contains ".")
        let dotIdx = accum.varName.IndexOf('.')
        let isStructField = dotIdx > 0 && dotIdx < accum.varName.Length - 1

        let initShape, currentShape =
            if isStructField then
                let baseName  = accum.varName.[..dotIdx - 1]
                let fieldName = accum.varName.[dotIdx + 1..]
                getStructField (Env.get preLoopEnv  baseName) fieldName,
                getStructField (Env.get postLoopEnv baseName) fieldName
            else
                Env.get preLoopEnv  accum.varName,
                Env.get postLoopEnv accum.varName

        if not (isMatrix initShape) || isUnknown initShape || isBottom initShape then ()
        else
            if not (isMatrix currentShape) then ()
            else
                // Self-reference check: for struct fields, check for FieldAccess mentions
                let selfRef =
                    if isStructField then
                        let baseName  = accum.varName.[..dotIdx - 1]
                        let fieldName = accum.varName.[dotIdx + 1..]
                        accum.deltaExprs |> List.exists (fun row ->
                            row |> List.exists (fun elem ->
                                exprMentionsFieldAccess elem baseName fieldName ||
                                exprMentionsVar elem accum.loopVar))
                    else
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
                            let isImprecise (d: Dim) = match d with Unknown | Range _ -> true | _ -> false
                            let refinedShapeOpt =
                                if accum.axis = Vert then
                                    if deltaRows_ = Unknown then None
                                    else
                                        let totalAdded = mulDim iterCount deltaRows_
                                        let refinedRows = addDim initRows totalAdded
                                        if isImprecise curRows then Some (Matrix(refinedRows, initCols))
                                        else None
                                else // Horz
                                    if deltaCols = Unknown then None
                                    else
                                        let totalAdded = mulDim iterCount deltaCols
                                        let refinedCols = addDim initCols totalAdded
                                        if isImprecise curCols then Some (Matrix(initRows, refinedCols))
                                        else None
                            match refinedShapeOpt with
                            | None -> ()
                            | Some refinedShape ->
                                if isStructField then
                                    let baseName  = accum.varName.[..dotIdx - 1]
                                    let fieldName = accum.varName.[dotIdx + 1..]
                                    let baseStruct = Env.get postLoopEnv baseName
                                    let updatedStruct = setStructField baseStruct fieldName refinedShape
                                    Env.set postLoopEnv baseName updatedStruct
                                else
                                    Env.set postLoopEnv accum.varName refinedShape
                        | _ -> ()


// ---------------------------------------------------------------------------
// Branch join helper
// ---------------------------------------------------------------------------

/// joinBranchResults: join analyzed branches, propagate if all returned.
let private joinBranchResults
    (env: Env)
    (ctx: AnalysisContext)
    (baselineConstraints: Set<string * string>)
    (baselineDimEquiv: DimEquiv.DimEquiv)
    (baselineRanges: Map<string, SharedTypes.Interval>)
    (branchEnvs: Env list)
    (branchConstraints: Set<string * string> list)
    (branchDimEquivs: DimEquiv.DimEquiv list)
    (branchRanges: Map<string, SharedTypes.Interval> list)
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
        let liveDimEquivs     = List.zip branchDimEquivs returnedFlags |> List.choose (fun (d, r) -> if not r then Some d else None)
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
            ctx.cst.constraints <- joinedConstraints

            // Provenance pruning: keep only provenance entries for surviving constraints
            ctx.cst.constraintProvenance <-
                joinedConstraints |> Set.fold (fun acc c ->
                    match Map.tryFind c ctx.cst.constraintProvenance with
                    | Some line -> Map.add c line acc
                    | None      -> acc) Map.empty

            // Intersect DimEquiv stores: keep equivalences present in ALL live branches
            let joinedDimEquiv =
                match liveDimEquivs with
                | [] -> baselineDimEquiv
                | first :: rest -> rest |> List.fold DimEquiv.intersect first
            ctx.cst.dimEquiv.parent.Clear()
            for kv in joinedDimEquiv.parent do ctx.cst.dimEquiv.parent.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.rank.Clear()
            for kv in joinedDimEquiv.rank do ctx.cst.dimEquiv.rank.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.concrete.Clear()
            for kv in joinedDimEquiv.concrete do ctx.cst.dimEquiv.concrete.[kv.Key] <- kv.Value

            // Join value_ranges
            ctx.cst.valueRanges <- joinValueRanges baselineRanges liveRanges


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
        let outputShapes = analyzeFunctionCall fname args line env warnings ctx 1
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
        let sig_ = ctx.call.nestedFunctionRegistry.[fname]
        if sig_.outputVars.IsEmpty then
            warnings.Add(warnProcedureInExpr line fname)
        let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx 1
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    elif ctx.ws.externalFunctions.ContainsKey(fname) then
        // External functions: no W_PROCEDURE_IN_EXPR (only applies to same-file functions)
        let extSig = ctx.ws.externalFunctions.[fname]
        let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx 1
        if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
    elif ctx.call.classRegistry.ContainsKey(fname) then
        // Constructor call: return Struct with declared properties
        let classInfo = ctx.call.classRegistry.[fname]
        if ctx.call.functionRegistry.ContainsKey(fname) then
            let outputShapes = analyzeFunctionCall fname args line env warnings ctx 1
            let baseShape = if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            match baseShape with
            | Struct(fields, isOpen) ->
                let fieldMap = Map.ofList fields
                let allFields =
                    classInfo.properties |> List.map (fun p ->
                        (p, defaultArg (Map.tryFind p fieldMap) UnknownShape))
                Struct(allFields, isOpen)
            | _ ->
                let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
                Struct(allFields, false)
        else
            let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
            Struct(allFields, false)
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
                let outputShapes = analyzeFunctionCall fname args line env warnings ctx 1
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
                let sig_ = ctx.call.nestedFunctionRegistry.[fname]
                if sig_.outputVars.IsEmpty then
                    warnings.Add(warnProcedureInExpr line fname)
                let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx 1
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            elif ctx.ws.externalFunctions.ContainsKey(fname) then
                // External functions: no W_PROCEDURE_IN_EXPR (only applies to same-file functions)
                let extSig = ctx.ws.externalFunctions.[fname]
                let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx 1
                if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
            elif ctx.call.classRegistry.ContainsKey(fname) then
                // Constructor call: return Struct with declared properties
                let classInfo = ctx.call.classRegistry.[fname]
                if ctx.call.functionRegistry.ContainsKey(fname) then
                    let outputShapes = analyzeFunctionCall fname args line env warnings ctx 1
                    let baseShape = if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
                    match baseShape with
                    | Struct(fields, isOpen) ->
                        let fieldMap = Map.ofList fields
                        let allFields =
                            classInfo.properties |> List.map (fun p ->
                                (p, defaultArg (Map.tryFind p fieldMap) UnknownShape))
                        Struct(allFields, isOpen)
                    | _ ->
                        let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
                        Struct(allFields, false)
                else
                    let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
                    Struct(allFields, false)
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

        // size() dimension aliasing: n = size(A, 1) => union(n, A.rowDim)
        // Records DimEquiv equivalence for backward propagation.
        // Also sets dimAliases for Symbolic source dims (so zeros(n,...) resolves inline).
        // For Concrete source dims, DimEquiv is set here; valueRanges is set after interval
        // tracking (below) so the concrete value takes precedence over the interval tracking
        // block, which would otherwise remove the entry (size() returns None from getExprInterval).
        let sizeConcreteExact : Interval option =
            match expr with
            | Apply(_, Var(_, "size"), [IndexExpr(_, argExpr); IndexExpr(_, dimArgExpr)]) ->
                let argVarName =
                    match argExpr with
                    | Var(_, n) -> Some n
                    | _ -> None
                let argShape =
                    match argVarName with
                    | Some n -> Env.get env n
                    | None -> UnknownShape
                let dimIdx = exprToDimIrCtx dimArgExpr env (Some ctx)
                let targetDim =
                    match argShape, dimIdx with
                    | Matrix(rowDim, _), Concrete 1 -> Some rowDim
                    | Matrix(_, colDim), Concrete 2 -> Some colDim
                    | _ -> None
                match targetDim with
                | Some dim when dim <> Unknown ->
                    let nameKey = name
                    let dimKey  = Shapes.dimStr dim
                    DimEquiv.union ctx.cst.dimEquiv nameKey dimKey |> ignore
                    match dim with
                    | Concrete n ->
                        // Concrete dim: record in DimEquiv now; valueRanges set after interval tracking.
                        DimEquiv.setConcrete ctx.cst.dimEquiv nameKey n |> ignore
                        Some { lo = Finite n; hi = Finite n }
                    | Symbolic _ ->
                        // Symbolic dim: set dimAliases so inline resolution works in builtins.
                        env.dimAliases <- Map.add name dim env.dimAliases
                        None
                    | _ -> None
                | _ -> None
            | _ -> None

        // Constraint validation: first binding of scalar to concrete value
        if isBottom oldShape && isScalar newShape then
            let concreteValue = tryExtractConstValue expr
            match concreteValue with
            | Some cv ->
                ctx.cst.scalarBindings <- Map.add name cv ctx.cst.scalarBindings
                validateBinding ctx env name cv warnings line
            | None -> ()

        // Interval tracking
        if isScalar newShape then
            let iv = getExprInterval expr env ctx
            match iv with
            | Some i ->
                ctx.cst.valueRanges <- Map.add name i ctx.cst.valueRanges
                Intervals.bridgeToDimEquiv ctx name i
            | None   -> ctx.cst.valueRanges <- Map.remove name ctx.cst.valueRanges
        else
            ctx.cst.valueRanges <- Map.remove name ctx.cst.valueRanges

        // Post-interval: apply concrete valueRanges from size() aliasing (takes precedence).
        // This runs after interval tracking so it is not overwritten by Map.remove.
        match sizeConcreteExact with
        | Some exact ->
            ctx.cst.valueRanges <- Map.add name exact ctx.cst.valueRanges
            Intervals.bridgeToDimEquiv ctx name exact
        | None -> ()

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

        let baselineRanges = ctx.cst.valueRanges
        let refinements = extractConditionRefinements cond env ctx
        applyRefinements ctx refinements false

        analyzeLoopBody body env warnings ctx None

        ctx.cst.valueRanges <- baselineRanges

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
                ctx.cst.valueRanges <- Map.add var_ { lo = Finite lo; hi = Finite hi } ctx.cst.valueRanges
            | _ ->
                let loDim = exprToDimIr left env
                let hiDim = exprToDimIr right env
                let loBound =
                    match loConst with
                    | Some v -> Finite v
                    | None -> match loDim with Concrete n -> Finite n | Symbolic s -> SymBound s | Range _ -> Unbounded | Unknown -> Unbounded
                let hiBound =
                    match hiConst with
                    | Some v -> Finite v
                    | None -> match hiDim with Concrete n -> Finite n | Symbolic s -> SymBound s | Range _ -> Unbounded | Unknown -> Unbounded
                ctx.cst.valueRanges <- Map.add var_ { lo = loBound; hi = hiBound } ctx.cst.valueRanges
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
        let baselineDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let baselineRanges = ctx.cst.valueRanges

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
        let thenDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let thenRanges = ctx.cst.valueRanges

        // Reset to baseline for else branch
        ctx.cst.constraints <- baselineConstraints
        ctx.cst.dimEquiv.parent.Clear()
        for kv in baselineDimEquiv.parent do ctx.cst.dimEquiv.parent.[kv.Key] <- kv.Value
        ctx.cst.dimEquiv.rank.Clear()
        for kv in baselineDimEquiv.rank do ctx.cst.dimEquiv.rank.[kv.Key] <- kv.Value
        ctx.cst.dimEquiv.concrete.Clear()
        for kv in baselineDimEquiv.concrete do ctx.cst.dimEquiv.concrete.[kv.Key] <- kv.Value
        ctx.cst.valueRanges <- baselineRanges

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
        let elseDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let elseRanges = ctx.cst.valueRanges

        joinBranchResults
            env ctx baselineConstraints baselineDimEquiv baselineRanges
            [thenEnv; elseEnv] [thenConstraints; elseConstraints]
            [thenDimEquiv; elseDimEquiv]
            [thenRanges; elseRanges] [thenReturned; elseReturned] None

    | IfChain({ line = line }, conditions, bodies, elseBody) ->
        for cond in conditions do wiredEvalExprFull cond env warnings ctx |> ignore

        let allRefinements = conditions |> List.map (fun cond -> extractConditionRefinements cond env ctx)
        let baselineConstraints = snapshotConstraints ctx
        let baselineDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let baselineRanges = ctx.cst.valueRanges

        let allBodies = bodies @ [ elseBody ]
        let mutable branchEnvs : Env list = []
        let mutable branchConstraints : Set<string * string> list = []
        let mutable branchDimEquivs : DimEquiv.DimEquiv list = []
        let mutable branchRanges : Map<string, SharedTypes.Interval> list = []
        let mutable returnedFlags : bool list = []
        let mutable deferredExc : System.Exception option = None

        for idx in 0 .. allBodies.Length - 1 do
            ctx.cst.constraints <- baselineConstraints
            ctx.cst.dimEquiv.parent.Clear()
            for kv in baselineDimEquiv.parent do ctx.cst.dimEquiv.parent.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.rank.Clear()
            for kv in baselineDimEquiv.rank do ctx.cst.dimEquiv.rank.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.concrete.Clear()
            for kv in baselineDimEquiv.concrete do ctx.cst.dimEquiv.concrete.[kv.Key] <- kv.Value
            ctx.cst.valueRanges <- baselineRanges

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
            branchDimEquivs <- branchDimEquivs @ [ DimEquiv.snapshot ctx.cst.dimEquiv ]
            branchRanges <- branchRanges @ [ ctx.cst.valueRanges ]
            returnedFlags <- returnedFlags @ [returned]

        joinBranchResults
            env ctx baselineConstraints baselineDimEquiv baselineRanges
            branchEnvs branchConstraints branchDimEquivs branchRanges returnedFlags deferredExc

    | Switch({ line = line }, expr, cases, otherwise) ->
        wiredEvalExprFull expr env warnings ctx |> ignore
        for (caseVal, _) in cases do wiredEvalExprFull caseVal env warnings ctx |> ignore

        // Extract the switch variable name for interval refinement (simple Var only).
        let switchVarName =
            match expr with
            | Ir.Var(_, name) -> Some name
            | _ -> None

        let baselineConstraints = snapshotConstraints ctx
        let baselineDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let baselineRanges = ctx.cst.valueRanges

        let allBodies = (cases |> List.map snd) @ [ otherwise ]
        let mutable branchEnvs : Env list = []
        let mutable branchConstraints : Set<string * string> list = []
        let mutable branchDimEquivs : DimEquiv.DimEquiv list = []
        let mutable branchRanges : Map<string, SharedTypes.Interval> list = []
        let mutable returnedFlags : bool list = []
        let mutable deferredExc : System.Exception option = None

        for caseIdx in 0 .. allBodies.Length - 1 do
            ctx.cst.constraints <- baselineConstraints
            ctx.cst.dimEquiv.parent.Clear()
            for kv in baselineDimEquiv.parent do ctx.cst.dimEquiv.parent.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.rank.Clear()
            for kv in baselineDimEquiv.rank do ctx.cst.dimEquiv.rank.[kv.Key] <- kv.Value
            ctx.cst.dimEquiv.concrete.Clear()
            for kv in baselineDimEquiv.concrete do ctx.cst.dimEquiv.concrete.[kv.Key] <- kv.Value
            ctx.cst.valueRanges <- baselineRanges

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
            branchDimEquivs <- branchDimEquivs @ [ DimEquiv.snapshot ctx.cst.dimEquiv ]
            branchRanges <- branchRanges @ [ ctx.cst.valueRanges ]
            returnedFlags <- returnedFlags @ [returned]

        joinBranchResults
            env ctx baselineConstraints baselineDimEquiv baselineRanges
            branchEnvs branchConstraints branchDimEquivs branchRanges returnedFlags deferredExc

    | Try(_, tryBody, catchBody) ->
        let baselineConstraints = snapshotConstraints ctx
        let baselineDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let baselineRanges = ctx.cst.valueRanges
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
        let tryDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let tryRanges = ctx.cst.valueRanges

        ctx.cst.constraints <- baselineConstraints
        ctx.cst.dimEquiv.parent.Clear()
        for kv in baselineDimEquiv.parent do ctx.cst.dimEquiv.parent.[kv.Key] <- kv.Value
        ctx.cst.dimEquiv.rank.Clear()
        for kv in baselineDimEquiv.rank do ctx.cst.dimEquiv.rank.[kv.Key] <- kv.Value
        ctx.cst.dimEquiv.concrete.Clear()
        for kv in baselineDimEquiv.concrete do ctx.cst.dimEquiv.concrete.[kv.Key] <- kv.Value
        ctx.cst.valueRanges <- baselineRanges

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
        let catchDimEquiv    = DimEquiv.snapshot ctx.cst.dimEquiv
        let catchRanges = ctx.cst.valueRanges

        joinBranchResults
            env ctx baselineConstraints baselineDimEquiv baselineRanges
            [tryEnv; catchEnv] [tryConstraints; catchConstraints]
            [tryDimEquiv; catchDimEquiv]
            [tryRanges; catchRanges] [tryReturned; catchReturned] deferredExc

    | OpaqueStmt({ line = line }, targets, raw) ->
        let firstWord = if raw.Trim() = "" then "" else raw.Trim().Split([| ' '; '\t'; ':' |]).[0]
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
                | IndexExpr(_, Var(_, varName)) ->
                    match Map.tryFind varName ctx.cst.valueRanges with
                    | Some iv when iv.lo = iv.hi ->
                        match iv.lo with
                        | SharedTypes.Finite k ->
                            let idx0 = k - 1
                            let currentElems = defaultArg elemMap Map.empty
                            let newElems = Map.add idx0 rhsShape currentElems
                            Env.set env baseName (Cell(rows, cols, Some newElems))
                        | _ ->
                            for arg in args do evalArgShapeHelper arg env warnings ctx baseShape |> ignore
                            Env.set env baseName (Cell(rows, cols, None))
                    | _ ->
                        for arg in args do evalArgShapeHelper arg env warnings ctx baseShape |> ignore
                        Env.set env baseName (Cell(rows, cols, None))
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
                // size() single-return aliasing (target = size(A, dim))
                if fname = "size" then
                    match args with
                    | [IndexExpr(_, argExpr); IndexExpr(_, dimArgExpr)] ->
                        let argShape =
                            match argExpr with
                            | Var(_, n) -> Env.get env n
                            | _ -> UnknownShape
                        let dimIdx = exprToDimIrCtx dimArgExpr env (Some ctx)
                        let targetDim =
                            match argShape, dimIdx with
                            | Matrix(rowDim, _), Concrete 1 -> Some rowDim
                            | Matrix(_, colDim), Concrete 2 -> Some colDim
                            | _ -> None
                        let tgt = targets.[0]
                        match targetDim with
                        | Some dim when dim <> Unknown && tgt <> "~" && not (tgt.Contains(".")) ->
                            DimEquiv.union ctx.cst.dimEquiv tgt (Shapes.dimStr dim) |> ignore
                            match dim with
                            | Concrete n ->
                                DimEquiv.setConcrete ctx.cst.dimEquiv tgt n |> ignore
                                let exact = { lo = Finite n; hi = Finite n }
                                ctx.cst.valueRanges <- Map.add tgt exact ctx.cst.valueRanges
                                Intervals.bridgeToDimEquiv ctx tgt exact
                            | Symbolic _ ->
                                env.dimAliases <- Map.add tgt dim env.dimAliases
                            | _ -> ()
                        | _ -> ()
                    | _ -> ()
            else
                match evalMultiBuiltinCall fname line numTargets args env warnings ctx wiredEvalExprFull wiredGetInterval with
                | Some shapes ->
                    for (target, shape) in List.zip targets shapes do bindTarget target shape
                    // [r, c] = size(A) aliasing
                    if fname = "size" && targets.Length = 2 then
                        match args with
                        | [IndexExpr(_, argExpr)] ->
                            let argShape =
                                match argExpr with
                                | Var(_, n) -> Env.get env n
                                | _ -> UnknownShape
                            match argShape with
                            | Matrix(rowDim, colDim) ->
                                let tgt0 = targets.[0]
                                let tgt1 = targets.[1]
                                if rowDim <> Unknown && tgt0 <> "~" && not (tgt0.Contains(".")) then
                                    DimEquiv.union ctx.cst.dimEquiv tgt0 (Shapes.dimStr rowDim) |> ignore
                                    match rowDim with
                                    | Concrete n ->
                                        DimEquiv.setConcrete ctx.cst.dimEquiv tgt0 n |> ignore
                                        let exact = { lo = Finite n; hi = Finite n }
                                        ctx.cst.valueRanges <- Map.add tgt0 exact ctx.cst.valueRanges
                                        Intervals.bridgeToDimEquiv ctx tgt0 exact
                                    | Symbolic _ ->
                                        env.dimAliases <- Map.add tgt0 rowDim env.dimAliases
                                    | _ -> ()
                                if colDim <> Unknown && tgt1 <> "~" && not (tgt1.Contains(".")) then
                                    DimEquiv.union ctx.cst.dimEquiv tgt1 (Shapes.dimStr colDim) |> ignore
                                    match colDim with
                                    | Concrete n ->
                                        DimEquiv.setConcrete ctx.cst.dimEquiv tgt1 n |> ignore
                                        let exact = { lo = Finite n; hi = Finite n }
                                        ctx.cst.valueRanges <- Map.add tgt1 exact ctx.cst.valueRanges
                                        Intervals.bridgeToDimEquiv ctx tgt1 exact
                                    | Symbolic _ ->
                                        env.dimAliases <- Map.add tgt1 colDim env.dimAliases
                                    | _ -> ()
                            | _ -> ()
                        | _ -> ()
                | None ->
                    let supported = defaultArg (Map.tryFind fname MULTI_SUPPORTED_FORMS) "unknown"
                    warnings.Add(warnMultiReturnCount line fname supported numTargets)
                    for target in targets do bindTarget target UnknownShape

        elif ctx.call.functionRegistry.ContainsKey(fname) then
            let outputShapes = analyzeFunctionCall fname args line env warnings ctx targets.Length
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        elif ctx.call.nestedFunctionRegistry.ContainsKey(fname) then
            let outputShapes = analyzeNestedFunctionCall fname args line env warnings ctx targets.Length
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        elif ctx.ws.externalFunctions.ContainsKey(fname) then
            let extSig = ctx.ws.externalFunctions.[fname]
            let outputShapes = analyzeExternalFunctionCall fname extSig args line env warnings ctx targets.Length
            if targets.Length <> outputShapes.Length then
                warnings.Add(warnMultiAssignCountMismatch line fname outputShapes.Length targets.Length)
                for target in targets do bindTarget target UnknownShape
            else
                for (target, shape) in List.zip targets outputShapes do bindTarget target shape

        elif ctx.call.classRegistry.ContainsKey(fname) then
            // Constructor call: return Struct with declared properties (single output)
            let classInfo = ctx.call.classRegistry.[fname]
            let structShape =
                if ctx.call.functionRegistry.ContainsKey(fname) then
                    let outputShapes = analyzeFunctionCall fname args line env warnings ctx targets.Length
                    let baseShape = if outputShapes.IsEmpty then UnknownShape else outputShapes.[0]
                    match baseShape with
                    | Struct(fields, isOpen) ->
                        let fieldMap = Map.ofList fields
                        let allFields =
                            classInfo.properties |> List.map (fun p ->
                                (p, defaultArg (Map.tryFind p fieldMap) UnknownShape))
                        Struct(allFields, isOpen)
                    | _ ->
                        let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
                        Struct(allFields, false)
                else
                    let allFields = classInfo.properties |> List.map (fun p -> (p, UnknownShape))
                    Struct(allFields, false)
            for target in targets do bindTarget target structShape

        else
            warnings.Add(warnUnknownFunction line fname)
            for target in targets do bindTarget target UnknownShape

    | _ ->
        warnings.Add(warnMultiAssignNonCall line)
        for target in targets do bindTarget target UnknownShape


// ---------------------------------------------------------------------------
// collectModifiedVars: syntactic scan of loop body for assigned variables.
// Over-approximates: any variable written anywhere in the body (including
// nested loops and branches) is considered modified.  Conservative is fine.
// ---------------------------------------------------------------------------

and private collectModifiedVars (body: Stmt list) : Set<string> =
    let mutable vars = Set.empty
    let rec scanStmt (s: Stmt) =
        match s with
        | Assign(_, name, _) -> vars <- Set.add name vars
        | StructAssign(_, baseName, _, _) -> vars <- Set.add baseName vars
        | CellAssign(_, baseName, _, _) -> vars <- Set.add baseName vars
        | IndexAssign(_, baseName, _, _) -> vars <- Set.add baseName vars
        | IndexStructAssign(_, baseName, _, _, _, _) -> vars <- Set.add baseName vars
        | FieldIndexAssign(_, baseName, _, _, _, _, _) -> vars <- Set.add baseName vars
        | AssignMulti(_, targets, _) ->
            for t in targets do if t <> "~" then vars <- Set.add t vars
        | For(_, var_, _, innerBody) ->
            vars <- Set.add var_ vars
            for s2 in innerBody do scanStmt s2
        | While(_, _, innerBody) ->
            for s2 in innerBody do scanStmt s2
        | If(_, _, thenBody, elseBody) ->
            for s2 in thenBody do scanStmt s2
            for s2 in elseBody do scanStmt s2
        | IfChain(_, _, bodies, elseBody) ->
            for branchBody in bodies do for s2 in branchBody do scanStmt s2
            for s2 in elseBody do scanStmt s2
        | Switch(_, _, cases, otherwise) ->
            for (_, caseBody) in cases do for s2 in caseBody do scanStmt s2
            for s2 in otherwise do scanStmt s2
        | Try(_, tryBody, catchBody) ->
            for s2 in tryBody do scanStmt s2
            for s2 in catchBody do scanStmt s2
        | _ -> ()
    for s in body do scanStmt s
    vars


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
        // Snapshot value ranges before loop body (persistent map: O(1) snapshot)
        let preLoopRanges = ctx.cst.valueRanges

        // Scope-limited widening: only widen variables actually assigned in the body.
        // Stable variables (assigned before loop, never reassigned inside) keep their
        // exact pre-loop intervals rather than being widened spuriously.
        let modifiedVars = collectModifiedVars body

        // Helper: restore pre-loop intervals for stable (non-modified) variables.
        // A stable variable is one that exists in preLoopRanges but was not modified
        // in the body. After widening/narrowing, we restore stable vars' original intervals.
        let restoreStableRanges () =
            ctx.cst.valueRanges <- ctx.cst.valueRanges |> Map.map (fun key iv ->
                let isModified = Set.contains key modifiedVars
                let isLoopVar  = Some key = loopVar
                if isModified || isLoopVar then iv
                else
                    match Map.tryFind key preLoopRanges with
                    | Some origIv -> origIv
                    | None -> iv)

        // Phase 1 (Discover)
        let preLoopEnv = Env.copy env
        try
            for s in body do analyzeStmtIr s env warnings ctx
        with
        | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()

        // Widen shapes
        let widened = widenEnv preLoopEnv env
        // Widen intervals in parallel with shape widening; then restore stable variables.
        ctx.cst.valueRanges <- widenValueRanges preLoopRanges ctx.cst.valueRanges loopVar
        restoreStableRanges ()

        // Phase 2 (Stabilize): Re-analyze if widening changed shapes OR intervals.
        // Scalar counters (count = count + 1) never change shape, so the old gate
        // (shapes-only) skipped Phase 2 entirely, leaving intervals under-widened.
        let shapesChanged = not (Env.localBindingsEqual env widened)
        let intervalsChanged = ctx.cst.valueRanges <> preLoopRanges
        if shapesChanged then
            Env.replaceLocal env widened
        if shapesChanged || intervalsChanged then
            try
                for s in body do analyzeStmtIr s env warnings ctx
            with
            | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()

        // Widen intervals again before post-loop join (mirrors finalWidened for shapes).
        // This second call widens from preLoopRanges (the pre-loop baseline), NOT from the
        // Phase-1-widened state.  With binary widening this would collapse Unbounded bounds
        // back to baseline (unsound), but with threshold widening the Phase-1 bounds are
        // still Finite, so the re-snap is "accidentally correct": the bound either stays at
        // the same threshold or moves to the next one.  If Phase 2 fires (shapes changed),
        // the body re-runs and the Phase-2 state is widened here; if Phase 2 does not fire
        // (scalar counters: shapes are always Scalar), this is the sole widening of the
        // Phase-1 intervals and the counter reaches at most the second threshold hop.
        ctx.cst.valueRanges <- widenValueRanges preLoopRanges ctx.cst.valueRanges loopVar
        restoreStableRanges ()

        // Phase 2.5 (Narrow): One narrowing pass to recover precision lost to widening.
        // Re-run the body with the stabilized (widened) state, then intersect the
        // resulting iterate with the current widened bounds.  Because we only do ONE
        // pass (not a fixpoint), termination is guaranteed.  Narrowing can only tighten
        // bounds, never widen them, so soundness is preserved.
        // Shape state (env) and warnings are restored after the pass: narrowing only
        // updates intervals, not shapes, and must not produce duplicate diagnostics.
        let preNarrowRanges = ctx.cst.valueRanges
        let preNarrowEnv = Env.copy env
        let throwawayWarnings = ResizeArray<Diagnostic>()
        try
            for s in body do analyzeStmtIr s env throwawayWarnings ctx
        with
        | :? EarlyReturn | :? EarlyBreak | :? EarlyContinue -> ()
        ctx.cst.valueRanges <- narrowValueRanges preNarrowRanges ctx.cst.valueRanges loopVar
        restoreStableRanges ()
        Env.replaceLocal env preNarrowEnv

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
    (numTargets: int)
    : Shape list =

    if not (ctx.call.functionRegistry.ContainsKey(funcName)) then [ UnknownShape ]
    else
        let sig_ = ctx.call.functionRegistry.[funcName]

        // Check argument count: too many args = error; too few = optional args (nargin support)
        if args.Length > sig_.parms.Length then
            warnings.Add(warnFunctionArgCountMismatch line funcName sig_.parms.Length args.Length)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        let actualArgCount = args.Length

        // Recursion guard
        if ctx.call.analyzingFunctions.Contains(funcName) then
            warnings.Add(warnRecursiveFunction line funcName)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        // Evaluate arg shapes for cache key (only for actually-passed args)
        let argShapes =
            args |> List.map (fun arg ->
                match arg with
                | IndexExpr(_, e) -> wiredEvalExprFull e env warnings ctx
                | _ -> UnknownShape)

        let argDimAliases =
            List.map2 (fun (param: string) arg ->
                match arg with
                | IndexExpr(_, e) -> (param, exprToDimIr e env)
                | _ -> (param, Unknown)) (List.take actualArgCount sig_.parms) args

        let cacheKey =
            funcName + ":n=" + string actualArgCount + ":o=" + string numTargets + ":(" +
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

                    // Bind passed parameters; leave extra params unbound (Bottom)
                    for (param, arg, argShape) in List.zip3 (List.take actualArgCount sig_.parms) args argShapes do
                        Env.set funcEnv param argShape
                        match arg with
                        | IndexExpr(_, e) ->
                            let callerDim = exprToDimIr e env
                            if callerDim <> Unknown then
                                funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                        | _ -> ()

                    // Inject nargin/nargout into function env
                    Env.set funcEnv "nargin" Scalar
                    ctx.cst.valueRanges <- Map.add "nargin" { lo = Finite actualArgCount; hi = Finite actualArgCount } ctx.cst.valueRanges
                    Env.set funcEnv "nargout" Scalar
                    ctx.cst.valueRanges <- Map.add "nargout" { lo = Finite numTargets; hi = Finite numTargets } ctx.cst.valueRanges

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
    (numTargets: int)
    : Shape list =

    if not (ctx.call.nestedFunctionRegistry.ContainsKey(funcName)) then [ UnknownShape ]
    else
        let sig_ = ctx.call.nestedFunctionRegistry.[funcName]

        // Too many args = error; too few = optional args (nargin support)
        if args.Length > sig_.parms.Length then
            warnings.Add(warnFunctionArgCountMismatch line funcName sig_.parms.Length args.Length)
            List.replicate (max sig_.outputVars.Length 1) UnknownShape
        else

        let actualArgCount = args.Length

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
                | _ -> (param, Unknown)) (List.take actualArgCount sig_.parms) args

        let cacheKey =
            "nested:" + funcName + ":n=" + string actualArgCount + ":o=" + string numTargets + ":(" +
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

                    // Bind passed parameters; leave extra params unbound (Bottom)
                    for (param, arg, argShape) in List.zip3 (List.take actualArgCount sig_.parms) args argShapes do
                        Env.set funcEnv param argShape
                        match arg with
                        | IndexExpr(_, e) ->
                            let callerDim = exprToDimIr e parentEnv
                            if callerDim <> Unknown then
                                funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                        | _ -> ()

                    // Inject nargin/nargout into function env
                    Env.set funcEnv "nargin" Scalar
                    ctx.cst.valueRanges <- Map.add "nargin" { lo = Finite actualArgCount; hi = Finite actualArgCount } ctx.cst.valueRanges
                    Env.set funcEnv "nargout" Scalar
                    ctx.cst.valueRanges <- Map.add "nargout" { lo = Finite numTargets; hi = Finite numTargets } ctx.cst.valueRanges

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
    (numTargets: int)
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

    // Too many args = error; too few = optional args (nargin support)
    if args.Length > primarySig.parms.Length then
        warnings.Add(warnFunctionArgCountMismatch line fname primarySig.parms.Length args.Length)
        List.replicate (max primarySig.outputVars.Length 1) UnknownShape
    else

    let actualArgCount = args.Length

    let argShapes =
        args |> List.map (fun arg ->
            match arg with
            | IndexExpr(_, e) -> wiredEvalExprFull e env warnings ctx
            | _ -> UnknownShape)

    let argDimAliases =
        List.map2 (fun (param: string) arg ->
            match arg with
            | IndexExpr(_, e) -> (param, exprToDimIr e env)
            | _ -> (param, Unknown)) (List.take actualArgCount primarySig.parms) args

    let cacheKey =
        "external:" + fname + ":n=" + string actualArgCount + ":o=" + string numTargets + ":(" +
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

                // Bind passed parameters; leave extra params unbound (Bottom)
                for (param, arg, argShape) in List.zip3 (List.take actualArgCount primarySig.parms) args argShapes do
                    Env.set funcEnv param argShape
                    match arg with
                    | IndexExpr(_, e) ->
                        let callerDim = exprToDimIr e env
                        if callerDim <> Unknown then
                            funcEnv.dimAliases <- Map.add param callerDim funcEnv.dimAliases
                    | _ -> ()

                // Inject nargin/nargout into function env
                Env.set funcEnv "nargin" Scalar
                ctx.cst.valueRanges <- Map.add "nargin" { lo = Finite actualArgCount; hi = Finite actualArgCount } ctx.cst.valueRanges
                Env.set funcEnv "nargout" Scalar
                ctx.cst.valueRanges <- Map.add "nargout" { lo = Finite numTargets; hi = Finite numTargets } ctx.cst.valueRanges

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
