module DimExtract

open Ir
open Shapes
open Env
open EndHelpers
open SharedTypes

// ---------------------------------------------------------------------------
// Dimension extraction from IR expressions.
// Port of analysis/dim_extract.py
// ---------------------------------------------------------------------------

/// exprToDimWithEnd: convert an expression (possibly containing End) to a Dim.
/// Substitutes End with endDim, enabling symbolic range extent computation.
let rec exprToDimWithEnd (expr: Expr) (env: Env) (endDim: Dim) : Dim =
    match expr with
    | End _ -> endDim
    | Const(_, _, v) ->
        if v = System.Math.Floor v && not (System.Double.IsInfinity v) then Concrete (int v)
        else Unknown
    | Var(_, _, name) ->
        // Check dim aliases first (propagates caller's dim name)
        match Map.tryFind name env.dimAliases with
        | Some d -> d
        | None   -> Symbolic (SymDim.SymDim.var name)
    | BinOp(_, _, op, left, right) ->
        let l = exprToDimWithEnd left env endDim
        let r = exprToDimWithEnd right env endDim
        match l, r with
        | Unknown, _ | _, Unknown -> Unknown
        | _ ->
            match op with
            | "+" -> addDim l r
            | "-" -> subDim l r
            | "*" -> mulDim l r
            | _   -> Unknown   // unsupported (/, .*, etc.)
    | _ -> Unknown


/// exprToDimIr: convert an expression to a Dim without End support.
/// When ctx (AnalysisContext) is provided, checks value_ranges for exact scalar values.
let rec exprToDimIr (expr: Expr) (env: Env) : Dim =
    exprToDimIrCtx expr env None

and exprToDimIrCtx (expr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : Dim =
    match expr with
    | Const(_, _, v) ->
        if v = System.Math.Floor v && not (System.Double.IsInfinity v) then Concrete (int v)
        else Unknown
    | End _ -> Unknown   // can't convert End without container context
    | Var(_, _, name) ->
        // Check dim aliases first
        match Map.tryFind name env.dimAliases with
        | Some d -> d
        | None ->
            // Check ctx.value_ranges for exact concrete value
            match ctx with
            | Some c ->
                match c.cst.valueRanges.TryGetValue(name) with
                | true, iv ->
                    match iv.lo, iv.hi with
                    | Finite lo, Finite hi when lo = hi -> Concrete lo
                    | _ -> Symbolic (SymDim.SymDim.var name)
                | false, _ -> Symbolic (SymDim.SymDim.var name)
            | None -> Symbolic (SymDim.SymDim.var name)
    | Neg _ -> Unknown   // Negative literal in dim context: treat as Unknown (mirrors Python)
    | BinOp(_, _, op, left, right) ->
        let l = exprToDimIrCtx left env ctx
        let r = exprToDimIrCtx right env ctx
        match l, r with
        | Unknown, _ | _, Unknown -> Unknown
        | _ ->
            match op with
            | "+" -> addDim l r
            | "-" -> subDim l r
            | "*" -> mulDim l r
            | _   -> Unknown   // unsupported operators
    | _ -> Unknown


/// extractIterationCount: extract iteration count from a for-loop iterator expression.
/// Handles BinOp(":", start, end) -> (end - start) + 1.
/// Returns Unknown for stepped ranges, non-range iterators, or unresolvable.
let extractIterationCount (itExpr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : Dim =
    match itExpr with
    | BinOp(_, _, ":", left, right) ->
        // Stepped range: BinOp(":", BinOp(":", start, step), end)
        match left with
        | BinOp(_, _, ":", _, _) -> Unknown  // stepped range
        | _ ->
            let a = exprToDimIrCtx left env ctx
            let b = exprToDimIrCtx right env ctx
            match a, b with
            | Unknown, _ | _, Unknown -> Unknown
            | Concrete a', Concrete b' -> Concrete (max 0 ((b' - a') + 1))
            | _ -> addDim (subDim b a) (Concrete 1)  // (b - a) + 1
    | _ -> Unknown


/// indexArgToExtentIr: return how many rows/cols an IndexArg selects.
/// Returns Unknown for Colon (caller must handle), None-mapped to Unknown here.
/// This partial port doesn't need warnings; the full version is in EvalExpr.fs.
let indexArgToExtentIr (arg: Ir.IndexArg) (env: Env) : Dim =
    match arg with
    | Colon _ -> Unknown
    | Range(_, _, startExpr, endExpr) ->
        let a = exprToDimIr startExpr env
        let b = exprToDimIr endExpr env
        match a, b with
        | Unknown, _ | _, Unknown -> Unknown
        | Concrete a', Concrete b' ->
            if b' < a' then Unknown
            else Concrete ((b' - a') + 1)
        | _ -> addDim (subDim b a) (Concrete 1)
    | IndexExpr _ -> Concrete 1
