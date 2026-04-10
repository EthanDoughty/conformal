// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Top-level analysis orchestrator. Runs a two-pass sweep over the IR:
// first to register function definitions, then to analyze the rest of
// the program against that registry.

module Analysis

open Ir
open Shapes
open Env
open Context
open Diagnostics
open StmtFuncAnalysis

/// Analyze a complete MATLAB program for shape consistency.
///
/// Two-pass analysis:
///   Pass 1 - register all FunctionDef statements into ctx.call.functionRegistry
///   Pass 2 - analyze all non-FunctionDef statements
///
/// Returns (final environment, deduplicated diagnostic list).
let analyzeProgramIr
    (program: Program)
    (ctx: AnalysisContext)
    : Env * Diagnostic list =

    let env = Env.create ()
    let warnings = ResizeArray<Diagnostic>()

    // Seed variables from % conformal:type directives
    for kv in ctx.typeAnnotations do
        Env.set env kv.Key kv.Value

    // Pass 1: register function definitions
    for item in program.body do
        match item with
        | FunctionDef({ line = line; col = col }, name, parms, outputVars, body, argAnns) ->
            ctx.call.functionRegistry.[name] <-
                { name = name; parms = parms; outputVars = outputVars; body = body
                  defLine = line; defCol = col; argShapes = argAnnotationsToShapes argAnns }
        | _ -> ()

    // Pass 1b: scan for classdef metadata in OpaqueStmts, populate classRegistry
    for item in program.body do
        match item with
        | OpaqueStmt(_, _, raw) when raw.StartsWith("classdef:") ->
            let parts = raw.Split(':')
            if parts.Length >= 2 then
                let className = parts.[1]
                let propNames =
                    if parts.Length >= 3 && parts.[2] <> "" then
                        parts.[2].Split(',') |> Array.toList
                    else []
                let superName =
                    if parts.Length >= 4 && parts.[3] <> "" then Some parts.[3] else None
                // Collect all registered functions as potential methods (safe over-approximation)
                let methodSigs =
                    ctx.call.functionRegistry
                    |> Seq.map (fun kv -> (kv.Key, kv.Value))
                    |> Map.ofSeq
                ctx.call.classRegistry.[className] <-
                    { name = className; properties = propNames; methods = methodSigs; superclass = superName }
        | _ -> ()

    // Seed known MATLAB base classes so inheritance chain walking doesn't break
    let knownBaseClasses = ["handle"; "matlab.unittest.TestCase"; "double"; "single"; "int32"; "uint32"; "int64"; "uint64"; "int16"; "uint16"; "int8"; "uint8"; "logical"; "char"]
    for name in knownBaseClasses do
        if not (ctx.call.classRegistry.ContainsKey(name)) then
            ctx.call.classRegistry.[name] <-
                { name = name; properties = []; methods = Map.empty; superclass = None }

    // Pass 2: analyze script statements (non-function bodies); top level eats all control flow
    for item in program.body do
        match item with
        | FunctionDef _ -> ()
        | _ -> analyzeStmtIr item env warnings ctx |> ignore

    // Post-analysis backward propagation pass: resolve symbolic dims via equivalence store.
    // This catches shapes assigned before the constraining operation ran (backward propagation).
    // Only resolves dims whose symbolic names are NOT bound as variables in the environment.
    // This prevents incorrectly concretizing symbolic dimension names that are also scalar variables.
    let allVars = env.bindings |> Map.toList |> List.map fst
    let allVarNames = env.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq
    for varName in allVars do
        let shape = Env.get env varName
        // Only resolve if not referencing a bound variable name
        let safeToResolve (d: Shapes.Dim) : bool =
            match d with
            | Shapes.Symbolic s ->
                // Check if any variable in this symbolic dim is bound in the env
                let vars = SymDim.SymDim.variables s
                Set.isEmpty (Set.intersect vars allVarNames)
            | _ -> true
        let resolved =
            match shape with
            | Shapes.Matrix(r, c) when safeToResolve r && safeToResolve c ->
                Constraints.resolveShape ctx shape
            | Shapes.Cell(r, c, elems) when safeToResolve r && safeToResolve c ->
                Constraints.resolveShape ctx shape
            | _ -> shape
        if resolved <> shape then
            Env.set env varName resolved

    // Post-analysis Coder pass: emit W_CODER_VARIABLE_SIZE and W_CODER_CELL_ARRAY
    // for top-level script variables with problematic shapes.
    // Scope: only variables in the top-level env (script vars + function return vars assigned here).
    if ctx.cst.coderMode then
        let isUnboundedDim (d: Shapes.Dim) : bool =
            match d with
            | Shapes.Unknown -> true
            | Shapes.Range(_, Shapes.BUnknown) -> true
            | Shapes.Range(Shapes.BUnknown, _) -> true
            | _ -> false
        for varName in (env.bindings |> Map.toList |> List.map fst) do
            let shape = Env.get env varName
            match shape with
            | Shapes.Matrix(r, c) when isUnboundedDim r || isUnboundedDim c ->
                warnings.Add(warnCoderVariableSize 0 varName shape)
            | Shapes.Cell _ ->
                warnings.Add(warnCoderCellArray 0 varName)
            | _ -> ()

    // Deduplicate warnings while preserving order
    let deduped = Seq.toList warnings |> List.distinctBy id
    (env, deduped)

/// Three-bucket shape coverage over the final top-level environment.
/// Returns (tracked, partial, untracked, total) where:
///   tracked   = concrete known shape (Scalar, Matrix(non-Unknown dims), Struct, Cell with known dims, etc.)
///   partial   = Matrix or Cell with at least one Unknown dim
///   untracked = UnknownShape
/// Excludes: Bottom entries, "nargin", "nargout", "ans", and names starting with "_".
let computeShapeCoverage (env: Env) : int * int * int * int =
    let excluded = Set.ofList ["nargin"; "nargout"; "ans"]
    let bindings =
        env.bindings
        |> Map.toList
        |> List.filter (fun (name, shape) ->
            shape <> Shapes.Bottom &&
            not (Set.contains name excluded) &&
            not (name.StartsWith("_")))
    let total = bindings.Length
    let untracked =
        bindings
        |> List.filter (fun (_, s) -> s = Shapes.UnknownShape)
        |> List.length
    let partial =
        bindings
        |> List.filter (fun (_, s) ->
            match s with
            | Shapes.Matrix(r, c) -> r = Shapes.Unknown || c = Shapes.Unknown
            | Shapes.Cell(r, c, _) -> r = Shapes.Unknown || c = Shapes.Unknown
            | _ -> false)
        |> List.length
    let tracked = total - untracked - partial
    (tracked, partial, untracked, total)
