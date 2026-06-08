// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Converts IR expressions into symbolic Dim values. Handles the end
// keyword by substituting the enclosing indexing extent, so expressions
// like A(end-1) produce a dim that correctly tracks the offset.

module DimExtract

open Ir
open Shapes
open Env
open EndHelpers
open SharedTypes

/// Convert an expression (possibly containing End) to a Dim.
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


/// Convert an expression to a Dim without End support.
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
                match Map.tryFind name c.cst.valueRanges with
                | Some iv ->
                    match iv.lo, iv.hi with
                    | Finite lo, Finite hi when lo = hi -> Concrete lo
                    | _ -> Symbolic (SymDim.SymDim.var name)
                | None -> Symbolic (SymDim.SymDim.var name)
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


/// Extract a concrete integer from an expression.
/// Handles Const and Neg(Const) to support negative step literals like -3 in a:(-3):b.
let tryExtractIntLiteral (expr: Expr) : int option =
    match expr with
    | Const(_, v) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) -> Some (int v)
    | Neg(_, Const(_, v)) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) -> Some (-(int v))
    | _ -> None


/// Fold an expression to a float constant using literals only (Const, Neg, BinOp +/-/*/).
/// Returns None for Var, function calls, End, or any non-literal node.
/// Returns None if the folded result is NaN or infinite (e.g. 1/0, 0/0).
let rec tryExtractConstFloat (expr: Expr) : float option =
    match expr with
    | Const(_, v) ->
        if System.Double.IsNaN v || System.Double.IsInfinity v then None
        else Some v
    | Neg(_, inner) ->
        match tryExtractConstFloat inner with
        | Some v -> Some (-v)
        | None   -> None
    | BinOp(_, op, left, right) ->
        match tryExtractConstFloat left, tryExtractConstFloat right with
        | Some l, Some r ->
            let result =
                match op with
                | "+" -> Some (l + r)
                | "-" -> Some (l - r)
                | "*" -> Some (l * r)
                | "/" -> Some (l / r)
                | _   -> None
            match result with
            | Some v when System.Double.IsNaN v || System.Double.IsInfinity v -> None
            | v -> v
        | _ -> None
    | _ -> None   // Var, End, Apply, FuncHandle, etc.


/// Fold an expression to a float constant, additionally resolving Var nodes
/// from integer valueRanges singletons in ctx. Mirrors tryExtractConstFloat
/// but also handles Var when ctx provides a singleton interval (Finite v, lo=hi).
/// Floats stored only as integer singletons for now (slice 1); float constants
/// (Lambda=0.2) still return None until slice 2 adds the float store.
let rec tryExtractConstFloatCtx (expr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : float option =
    match expr with
    | Const(_, v) ->
        if System.Double.IsNaN v || System.Double.IsInfinity v then None
        else Some v
    | Var(_, name) ->
        match ctx with
        | Some c ->
            match Map.tryFind name c.cst.valueRanges with
            | Some iv ->
                match iv.lo, iv.hi with
                | Finite lo, Finite hi when lo = hi -> Some (float lo)
                | _ -> None
            | None -> None
        | None -> None
    | Neg(_, inner) ->
        match tryExtractConstFloatCtx inner env ctx with
        | Some v -> Some (-v)
        | None   -> None
    | BinOp(_, op, left, right) ->
        match tryExtractConstFloatCtx left env ctx, tryExtractConstFloatCtx right env ctx with
        | Some l, Some r ->
            let result =
                match op with
                | "+" -> Some (l + r)
                | "-" -> Some (l - r)
                | "*" -> Some (l * r)
                | "/" -> Some (l / r)
                | _   -> None
            match result with
            | Some v when System.Double.IsNaN v || System.Double.IsInfinity v -> None
            | v -> v
        | _ -> None
    | _ -> None   // End, Apply, FuncHandle, etc.


// Half-away-from-zero rounding: sign(x)*floor(|x|+0.5).
// Avoids System.Math.Round (banker's) and JS Math.round (half-up toward +inf).
// Identical for native .NET and Fable because it uses only floor/abs/sign/arithmetic.
let private roundHalfAwayFromZero (x: float) : float =
    let s = if x < 0.0 then -1.0 elif x > 0.0 then 1.0 else 0.0
    s * System.Math.Floor(System.Math.Abs(x) + 0.5)


/// Compute stepped-range length using Cleve Moler's colonop algorithm.
/// Inputs are folded IEEE-754 doubles from literals.
/// Returns Concrete len (>= 0) or Unknown when the length cannot be determined safely.
let steppedRangeLengthFloat (a: float) (step: float) (b: float) : Dim =
    // Guard: non-finite inputs or zero step.
    if System.Double.IsNaN a || System.Double.IsInfinity a ||
       System.Double.IsNaN step || System.Double.IsInfinity step || step = 0.0 ||
       System.Double.IsNaN b || System.Double.IsInfinity b then
        Unknown
    else
        let sgn  = if step > 0.0 then 1.0 elif step < 0.0 then -1.0 else 0.0
        let mutable n = roundHalfAwayFromZero ((b - a) / step) |> int
        let eps  = 2.220446049250313e-16   // 2^-52
        let tol  = 2.0 * eps * (max (System.Math.Abs a) (System.Math.Abs b))
        if   sgn * (a + float n * step - b) >  tol then n <- n - 1
        elif sgn * (a + float (n + 1) * step - b) <= -tol then n <- n + 1
        let len = if n < 0 then 0 else n + 1
        Concrete len


/// Extract iteration count from a for-loop iterator expression.
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


/// Return how many rows/cols an IndexArg selects.
/// Returns Unknown for Colon (caller must handle). The full version is in EvalExpr.fs.
let indexArgToExtentIr (arg: Ir.IndexArg) (env: Env) : Dim =
    match arg with
    | Colon _ -> Unknown
    | Ir.Range(_, startExpr, endExpr) ->
        let a = exprToDimIr startExpr env
        let b = exprToDimIr endExpr env
        match a, b with
        | Unknown, _ | _, Unknown -> Unknown
        | Concrete a', Concrete b' ->
            if b' < a' then Unknown
            else Concrete ((b' - a') + 1)
        | _ -> addDim (subDim b a) (Concrete 1)
    | Ir.SteppedRange(_, startExpr, stepExpr, endExpr) ->
        let stepInt = tryExtractIntLiteral stepExpr
        let a = exprToDimIr startExpr env
        let b = exprToDimIr endExpr env
        match stepInt with
        | Some s when s > 0 ->
            match a, b with
            | Concrete a', Concrete b' ->
                if b' < a' then Concrete 0
                else Concrete ((b' - a') / s + 1)
            | _ -> Unknown
        | Some s when s < 0 ->
            match a, b with
            | Concrete a', Concrete b' ->
                if a' < b' then Concrete 0
                else Concrete ((a' - b') / (-s) + 1)
            | _ -> Unknown
        | _ -> Unknown
    | IndexExpr _ -> Concrete 1
