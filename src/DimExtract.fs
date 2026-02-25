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
    | Const(_, v) ->
        if v = System.Math.Floor(v : float) && not (System.Double.IsInfinity v) then Concrete (int v)
        else Unknown
    | Var(_, name) ->
        // Check dim aliases first (propagates caller's dim name)
        match Map.tryFind name env.dimAliases with
        | Some d -> d
        | None   -> Symbolic (SymDim.SymDim.var name)
    | BinOp(_, op, left, right) ->
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
    | Const(_, v) ->
        if v = System.Math.Floor(v : float) && not (System.Double.IsInfinity v) then Concrete (int v)
        else Unknown
    | End _ -> Unknown   // can't convert End without container context
    | Var(_, name) ->
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
    | BinOp(_, op, left, right) ->
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


/// tryExtractIntLiteral: extract a concrete integer from an expression.
/// Handles Const and Neg(Const) to support negative step literals like -3 in a:(-3):b.
let private tryExtractIntLiteral (expr: Expr) : int option =
    match expr with
    | Const(_, v) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) -> Some (int v)
    | Neg(_, Const(_, v)) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) -> Some (-(int v))
    | _ -> None


/// extractIterationCount: extract iteration count from a for-loop iterator expression.
///
/// Handles three structural forms (as emitted by MatlabParser):
///
///  1. Non-stepped:    BinOp(":", a, b)
///     Count = (b - a) + 1; symbolic ok.
///
///  2. Positive step (parenthesized):  BinOp(":", BinOp(":", a, step), b)
///     where step > 0.
///     Count = floor((b-a)/step)+1 for concrete a,b; Unknown for symbolic.
///
///  3. Negative step (unparenthesized):  BinOp(":", a, Neg(BinOp(":", s, b)))
///     Parser produces this form for `a:-s:b` (e.g. 10:-3:1 -> BinOp(":", 10, Neg(BinOp(":", 3, 1)))).
///     Effective step = -s (negative), count = floor((a-b)/s)+1 for concrete a,b.
///
/// Returns Unknown for non-range iterators, symbolic steps, or unresolvable.
let extractIterationCount (itExpr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : Dim =
    match itExpr with
    | BinOp(_, ":", left, right) ->
        match left, right with
        // Form 2: a:step:b (parenthesized positive or explicit step)
        // BinOp(":", BinOp(":", a, step), b)
        | BinOp(_, ":", startExpr, stepExpr), bExpr ->
            let stepInt = tryExtractIntLiteral stepExpr
            let a = exprToDimIrCtx startExpr env ctx
            let b = exprToDimIrCtx bExpr      env ctx
            match stepInt with
            | Some 1 ->
                match a, b with
                | Unknown, _ | _, Unknown -> Unknown
                | Concrete a', Concrete b' -> Concrete (max 0 ((b' - a') + 1))
                | _ -> addDim (subDim b a) (Concrete 1)
            | Some -1 ->
                match a, b with
                | Unknown, _ | _, Unknown -> Unknown
                | Concrete a', Concrete b' -> Concrete (max 0 ((a' - b') + 1))
                | _ -> addDim (subDim a b) (Concrete 1)
            | Some step' when step' > 0 ->
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = b' - a'
                    if dividend < 0 then Concrete 0
                    else Concrete (dividend / step' + 1)
                | _ -> Unknown
            | Some step' when step' < 0 ->
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = a' - b'
                    if dividend < 0 then Concrete 0
                    else Concrete (dividend / (-step') + 1)
                | _ -> Unknown
            | _ -> Unknown

        // Form 3: a:-s:b (unparenthesized negative step)
        // Parser produces BinOp(":", a, Neg(BinOp(":", s, b)))
        // because -s:b is parsed as Neg(BinOp(":", s, b))
        | aExpr, Neg(_, BinOp(_, ":", stepExpr, bExpr)) ->
            let stepInt = tryExtractIntLiteral stepExpr
            let a = exprToDimIrCtx aExpr env ctx
            let b = exprToDimIrCtx bExpr  env ctx
            match stepInt with
            | Some 1 ->
                // Effective step=-1: count = (a-b)+1
                match a, b with
                | Unknown, _ | _, Unknown -> Unknown
                | Concrete a', Concrete b' -> Concrete (max 0 ((a' - b') + 1))
                | _ -> addDim (subDim a b) (Concrete 1)
            | Some s when s > 0 ->
                // Effective step=-s: count = floor((a-b)/s)+1
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = a' - b'
                    if dividend < 0 then Concrete 0
                    else Concrete (dividend / s + 1)
                | _ -> Unknown
            | _ -> Unknown

        // Form 1: non-stepped a:b
        | aExpr, bExpr ->
            let a = exprToDimIrCtx aExpr env ctx
            let b = exprToDimIrCtx bExpr env ctx
            match a, b with
            | Unknown, _ | _, Unknown -> Unknown
            | Concrete a', Concrete b' -> Concrete (max 0 ((b' - a') + 1))
            | _ -> addDim (subDim b a) (Concrete 1)
    | _ -> Unknown


/// indexArgToExtentIr: return how many rows/cols an IndexArg selects.
/// Returns Unknown for Colon (caller must handle), None-mapped to Unknown here.
/// This partial port doesn't need warnings; the full version is in EvalExpr.fs.
let indexArgToExtentIr (arg: Ir.IndexArg) (env: Env) : Dim =
    match arg with
    | Colon _ -> Unknown
    | Range(_, startExpr, endExpr) ->
        let a = exprToDimIr startExpr env
        let b = exprToDimIr endExpr env
        match a, b with
        | Unknown, _ | _, Unknown -> Unknown
        | Concrete a', Concrete b' ->
            if b' < a' then Unknown
            else Concrete ((b' - a') + 1)
        | _ -> addDim (subDim b a) (Concrete 1)
    | IndexExpr _ -> Concrete 1
