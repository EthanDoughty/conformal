module Analysis

open Ir
open Shapes
open Env
open Context
open Diagnostics
open StmtFuncAnalysis

// ---------------------------------------------------------------------------
// Analysis entry point.
// Port of analysis/__init__.py  analyze_program_ir().
// ---------------------------------------------------------------------------

/// analyzeProgramIr: analyze a complete MATLAB program for shape consistency.
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

    // Pass 1: register function definitions
    for item in program.body do
        match item with
        | FunctionDef({ line = line; col = col }, name, parms, outputVars, body) ->
            ctx.call.functionRegistry.[name] <-
                { name = name; parms = parms; outputVars = outputVars; body = body; defLine = line; defCol = col }
        | _ -> ()

    // Pass 2: analyze script statements (non-function bodies)
    try
        for item in program.body do
            match item with
            | FunctionDef _ -> ()
            | _ -> analyzeStmtIr item env warnings ctx
    with
    | :? EarlyReturn     -> ()
    | :? EarlyBreak      -> ()
    | :? EarlyContinue   -> ()

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

    // Deduplicate warnings while preserving order
    let deduped = Seq.toList warnings |> List.distinctBy id
    (env, deduped)
