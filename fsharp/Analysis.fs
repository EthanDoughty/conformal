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
    let warnings = ref []

    // Pass 1: register function definitions
    for item in program.body do
        match item with
        | FunctionDef(_, _, name, parms, outputVars, body) ->
            ctx.call.functionRegistry.[name] <-
                { name = name; parms = parms; outputVars = outputVars; body = body }
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

    // Deduplicate warnings while preserving order
    let deduped = warnings.Value |> List.distinctBy id
    (env, deduped)
