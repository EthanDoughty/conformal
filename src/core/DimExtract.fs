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
        // Guard: non-finite or outside Int32 range must degrade to Unknown to prevent overflow.
        if System.Double.IsNaN v || System.Double.IsInfinity v then Unknown
        elif v = System.Math.Floor(v : float) && v >= float System.Int32.MinValue && v <= float System.Int32.MaxValue then
            Concrete (int v)
        else Unknown
    | Var(_, name) ->
        if name = "Inf" || name = "inf" || name = "NaN" || name = "nan" then Unknown
        else
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


/// Builtins whose value is a single extent derived from an argument's shape.
let SHAPE_QUERY_BUILTINS : Set<string> = Set.ofList ["numel"; "length"; "size"]

/// Row and column extents of a shape, or None when the shape carries none.
/// Struct is excluded on purpose: a struct array's element count is not modeled.
let shapeExtents (s: Shape) : (Dim * Dim) option =
    match s with
    | Scalar        -> Some (Concrete 1, Concrete 1)
    | Matrix(r, c)  -> Some (r, c)
    | Cell(r, c, _) -> Some (r, c)
    | _             -> None

/// True when either extent is a non-point dim or a literal zero.  Zero is
/// excluded because an empty-literal accumulator (K = {}; A = []) keeps a zero
/// extent that MATLAB would have grown, and folding it turns a sound unknown
/// into a confident wrong answer.
let private extentsUnfoldable (r: Dim) (c: Dim) =
    match r, c with
    | Unknown, _ | _, Unknown | Range _, _ | _, Range _ -> true
    | Concrete 0, _ | _, Concrete 0 -> true
    | _ -> false

let numelDim (s: Shape) : Dim =
    match shapeExtents s with
    | Some (r, c) when not (extentsUnfoldable r c) -> mulDim r c
    | _ -> Unknown

let lengthDim (s: Shape) : Dim =
    match shapeExtents s with
    | Some (r, c) when not (extentsUnfoldable r c) ->
        match r, c with
        | Concrete 1, d | d, Concrete 1 -> d
        | Concrete m, Concrete n -> Concrete (max m n)
        | _ -> Unknown            // max of two symbolics is not representable
    | _ -> Unknown


/// Convert an expression to a Dim without End support.
/// When ctx (AnalysisContext) is provided, checks value_ranges for exact scalar values.
let rec exprToDimIr (expr: Expr) (env: Env) : Dim =
    exprToDimIrCtx expr env None

and exprToDimIrCtx (expr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : Dim =
    match expr with
    | Const(_, v) ->
        // Guard: non-finite or outside Int32 range must degrade to Unknown to prevent overflow.
        if System.Double.IsNaN v || System.Double.IsInfinity v then Unknown
        elif v = System.Math.Floor(v : float) && v >= float System.Int32.MinValue && v <= float System.Int32.MaxValue then
            Concrete (int v)
        else Unknown
    | End _ -> Unknown   // can't convert End without container context
    | Var(_, name) ->
        // Special names that represent non-finite values: treat as Unknown to prevent
        // symbolic dims like (Inf + 1) or (NaN + 1) from entering the shape lattice.
        if name = "Inf" || name = "inf" || name = "NaN" || name = "nan" then Unknown
        else
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
    // A shape query written inline in a dimension argument.  Restricted to a
    // bare variable argument so the shape can be read from env without
    // evaluating (DimExtract precedes EvalExpr in compile order), which is the
    // same restriction the statement-level hook in StmtFuncAnalysis already uses.
    | Apply(_, Var(_, fname), fnArgs) when Set.contains fname SHAPE_QUERY_BUILTINS ->
        match ctx with
        | None -> Unknown          // no shadowing information available: stay conservative
        | Some c ->
            let shadowed =
                c.call.functionRegistry.ContainsKey fname
                || c.call.nestedFunctionRegistry.ContainsKey fname
                || c.ws.privateFunctions.ContainsKey fname
                || c.ws.externalFunctions.ContainsKey fname
                || (match Env.get env fname with FunctionHandle _ -> true | _ -> false)
            if shadowed then Unknown
            else
                match fname, fnArgs with
                | "numel",  [IndexExpr(_, Var(_, v))] -> numelDim  (Env.get env v)
                | "length", [IndexExpr(_, Var(_, v))] -> lengthDim (Env.get env v)
                | "size",   [IndexExpr(_, Var(_, v)); IndexExpr(_, kExpr)] ->
                    match shapeExtents (Env.get env v), exprToDimIrCtx kExpr env ctx with
                    | Some (r, _), Concrete 1 when r <> Unknown -> r
                    | Some (_, cD), Concrete 2 when cD <> Unknown -> cD
                    | _ -> Unknown      // k >= 3 is left to the ndArraySlices path
                | _ -> Unknown
    | _ -> Unknown


/// Extract a concrete integer from an expression.
/// Handles Const and Neg(Const) to support negative step literals like -3 in a:(-3):b.
/// Returns None if the value is non-finite or outside the Int32 range to prevent overflow.
let tryExtractIntLiteral (expr: Expr) : int option =
    let inline isInt32Range (v: float) = v >= float System.Int32.MinValue && v <= float System.Int32.MaxValue
    match expr with
    | Const(_, v) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) && not (System.Double.IsNaN v) && isInt32Range v ->
        Some (int v)
    | Neg(_, Const(_, v)) when v = System.Math.Floor(v) && not (System.Double.IsInfinity v) && not (System.Double.IsNaN v) && isInt32Range v ->
        Some (-(int v))
    | _ -> None


/// Fold an expression to a float constant using literals only (Const, Neg, and
/// BinOp with the four arithmetic operators).
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
/// from the trusted-constant store in ctx. Resolves only from trustedConsts
/// (decoupled from valueRanges and joins by construction). Enables chains like
/// b = a/2 as long as both a and b have trusted-constant entries.
let rec tryExtractConstFloatCtx (expr: Expr) (env: Env) (ctx: Context.AnalysisContext option) : float option =
    match expr with
    | Const(_, v) ->
        if System.Double.IsNaN v || System.Double.IsInfinity v then None
        else Some v
    | Var(_, name) ->
        match ctx with
        | Some c -> Map.tryFind name c.cst.trustedConsts
        | None   -> None
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
        let rawN = roundHalfAwayFromZero ((b - a) / step)
        // Overflow guard: reject non-finite or out-of-Int32 range before converting.
        // Use strict < 2147483647.0 so that values at the boundary (e.g. 0:1:2147483646
        // where rawN=2147483646) pass, but rawN=2147483647 (-> len=2147483648) is rejected.
        if System.Double.IsNaN rawN || System.Double.IsInfinity rawN
           || rawN >= 2147483647.0 || rawN < -2147483647.0 then
            Unknown
        else
        // Use int64 for n to avoid Int32 overflow in the n+1 tol-correction path.
        let mutable n = int64 rawN
        let eps  = 2.220446049250313e-16   // 2^-52
        let tol  = 2.0 * eps * (max (System.Math.Abs a) (System.Math.Abs b))
        if   sgn * (a + float n * step - b) >  tol then n <- n - 1L
        elif sgn * (a + float (n + 1L) * step - b) <= -tol then n <- n + 1L
        let len = if n < 0L then 0L else n + 1L
        // Guard the final length against overflow before converting to int.
        if len < 0L || len > 2147483647L then Unknown
        else Concrete (int len)


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
                | Concrete a', Concrete b' ->
                    let len64 = int64 b' - int64 a' + 1L
                    if len64 < 0L || len64 > 2147483647L then Unknown
                    else Concrete (max 0 (int len64))
                | _ -> addDim (subDim b a) (Concrete 1)
            | Some -1 ->
                match a, b with
                | Unknown, _ | _, Unknown -> Unknown
                | Concrete a', Concrete b' ->
                    let len64 = int64 a' - int64 b' + 1L
                    if len64 < 0L || len64 > 2147483647L then Unknown
                    else Concrete (max 0 (int len64))
                | _ -> addDim (subDim a b) (Concrete 1)
            | Some step' when step' > 0 ->
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = int64 b' - int64 a'
                    if dividend < 0L then Concrete 0
                    else
                        let len64 = dividend / int64 step' + 1L
                        if len64 > 2147483647L then Unknown else Concrete (int len64)
                | _ -> Unknown
            | Some step' when step' < 0 ->
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = int64 a' - int64 b'
                    if dividend < 0L then Concrete 0
                    else
                        let len64 = dividend / int64 (-step') + 1L
                        if len64 > 2147483647L then Unknown else Concrete (int len64)
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
                | Concrete a', Concrete b' ->
                    let len64 = int64 a' - int64 b' + 1L
                    if len64 < 0L || len64 > 2147483647L then Unknown
                    else Concrete (max 0 (int len64))
                | _ -> addDim (subDim a b) (Concrete 1)
            | Some s when s > 0 ->
                // Effective step=-s: count = floor((a-b)/s)+1
                match a, b with
                | Concrete a', Concrete b' ->
                    let dividend = int64 a' - int64 b'
                    if dividend < 0L then Concrete 0
                    else
                        let len64 = dividend / int64 s + 1L
                        if len64 > 2147483647L then Unknown else Concrete (int len64)
                | _ -> Unknown
            | _ -> Unknown

        // Form 1: non-stepped a:b
        | aExpr, bExpr ->
            let a = exprToDimIrCtx aExpr env ctx
            let b = exprToDimIrCtx bExpr env ctx
            match a, b with
            | Unknown, _ | _, Unknown -> Unknown
            | Concrete a', Concrete b' ->
                let len64 = int64 b' - int64 a' + 1L
                if len64 < 0L || len64 > 2147483647L then Unknown
                else Concrete (max 0 (int len64))
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
