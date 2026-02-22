module Intervals

open SymDim

// ---------------------------------------------------------------------------
// Integer interval abstract domain for value range analysis.
// Port of analysis/intervals.py
// ---------------------------------------------------------------------------

/// IntervalBound: concrete int, symbolic SymDim, or unbounded (None)
type IntervalBound =
    | Finite    of int
    | SymBound  of SymDim
    | Unbounded   // represents -infinity (lo) or +infinity (hi)


/// Interval: [lo, hi] with optional symbolic bounds.
/// lo=Unbounded means -infinity, hi=Unbounded means +infinity.
/// Invariant: if both concrete (Finite), lo <= hi.
[<Struct>]
type Interval = {
    lo: IntervalBound
    hi: IntervalBound
}

let makeInterval (lo: IntervalBound) (hi: IntervalBound) : Interval =
    // Guard: only validate if both concrete ints
    match lo, hi with
    | Finite l, Finite h when l > h ->
        failwith ("Invalid interval: lo=" + string l + " > hi=" + string h)
    | _ -> ()
    { lo = lo; hi = hi }

let intervalUnbounded () : Interval = { lo = Unbounded; hi = Unbounded }

/// isConcreteB: check if a bound is a concrete int (not SymDim)
let private isConcreteB (b: IntervalBound) : bool =
    match b with
    | Finite _ | Unbounded -> true
    | SymBound _ -> false


/// joinInterval: convex hull of two optional intervals (lattice join).
/// None represents absence of information.
let joinInterval (a: Interval option) (b: Interval option) : Interval option =
    match a, b with
    | None, _ -> b
    | _, None -> a
    | Some ia, Some ib ->
        // min of lower bounds (Unbounded = -infinity)
        let newLo =
            match ia.lo, ib.lo with
            | Unbounded, _ | _, Unbounded -> Unbounded
            | Finite la, Finite lb -> Finite (min la lb)
            | lo1, lo2 when lo1 = lo2 -> lo1   // structural equality (works for SymBound)
            | _ -> Unbounded  // can't compare: widen to -inf (sound)

        // max of upper bounds (Unbounded = +infinity)
        let newHi =
            match ia.hi, ib.hi with
            | Unbounded, _ | _, Unbounded -> Unbounded
            | Finite ha, Finite hb -> Finite (max ha hb)
            | hi1, hi2 when hi1 = hi2 -> hi1
            | _ -> Unbounded  // can't compare: widen to +inf (sound)

        Some { lo = newLo; hi = newHi }


/// widenInterval: push bounds to infinity when they move.
let widenInterval (oldIv: Interval) (newIv: Interval) : Interval =
    let lo =
        match newIv.lo, oldIv.lo with
        | Unbounded, _ | _, Unbounded -> oldIv.lo
        | Finite nl, Finite ol ->
            if nl < ol then Unbounded else oldIv.lo
        | _ -> Unbounded   // symbolic: can't prove it didn't move, widen (sound)

    let hi =
        match newIv.hi, oldIv.hi with
        | Unbounded, _ | _, Unbounded -> oldIv.hi
        | Finite nh, Finite oh ->
            if nh > oh then Unbounded else oldIv.hi
        | _ -> Unbounded   // symbolic: can't prove it didn't move, widen (sound)

    { lo = lo; hi = hi }


/// meetInterval: intersection of two intervals (lattice meet).
/// Returns None if intervals don't overlap.
let meetInterval (a: Interval) (b: Interval) : Interval option =
    // max of lower bounds
    let newLo =
        match a.lo, b.lo with
        | Unbounded, _ -> b.lo
        | _, Unbounded -> a.lo
        | Finite la, Finite lb -> Finite (max la lb)
        | lo1, lo2 when lo1 = lo2 -> lo1
        | _ -> Unbounded   // can't compare: conservative

    // min of upper bounds
    let newHi =
        match a.hi, b.hi with
        | Unbounded, _ -> b.hi
        | _, Unbounded -> a.hi
        | Finite ha, Finite hb -> Finite (min ha hb)
        | hi1, hi2 when hi1 = hi2 -> hi1
        | _ -> Unbounded   // can't compare: conservative

    // Check validity (non-empty): only for concrete int bounds
    match newLo, newHi with
    | Finite l, Finite h when l > h -> None   // empty interval
    | _ -> Some { lo = newLo; hi = newHi }


/// intervalAdd: [a,b] + [c,d] = [a+c, b+d].
/// Returns None (top) if any bound is symbolic.
let intervalAdd (a: Interval option) (b: Interval option) : Interval option =
    match a, b with
    | None, _ | _, None -> None
    | Some ia, Some ib ->
        // Short-circuit on symbolic bounds
        if not (isConcreteB ia.lo && isConcreteB ia.hi && isConcreteB ib.lo && isConcreteB ib.hi) then
            None
        else
            let newLo =
                match ia.lo, ib.lo with
                | Unbounded, _ | _, Unbounded -> Unbounded
                | Finite l1, Finite l2 -> Finite (l1 + l2)
                | _ -> Unbounded
            let newHi =
                match ia.hi, ib.hi with
                | Unbounded, _ | _, Unbounded -> Unbounded
                | Finite h1, Finite h2 -> Finite (h1 + h2)
                | _ -> Unbounded
            Some { lo = newLo; hi = newHi }


/// intervalSub: [a,b] - [c,d] = [a-d, b-c].
/// Returns None (top) if any bound is symbolic.
let intervalSub (a: Interval option) (b: Interval option) : Interval option =
    match a, b with
    | None, _ | _, None -> None
    | Some ia, Some ib ->
        if not (isConcreteB ia.lo && isConcreteB ia.hi && isConcreteB ib.lo && isConcreteB ib.hi) then
            None
        else
            let newLo =
                match ia.lo, ib.hi with
                | Unbounded, _ | _, Unbounded -> Unbounded
                | Finite l, Finite h -> Finite (l - h)   // a.lo - b.hi
                | _ -> Unbounded
            let newHi =
                match ia.hi, ib.lo with
                | Unbounded, _ | _, Unbounded -> Unbounded
                | Finite h, Finite l -> Finite (h - l)   // a.hi - b.lo
                | _ -> Unbounded
            Some { lo = newLo; hi = newHi }


/// intervalMul: standard 4-product min/max.
/// Returns None (top) if any bound is symbolic.
let intervalMul (a: Interval option) (b: Interval option) : Interval option =
    match a, b with
    | None, _ | _, None -> None
    | Some ia, Some ib ->
        if not (isConcreteB ia.lo && isConcreteB ia.hi && isConcreteB ib.lo && isConcreteB ib.hi) then
            None
        else
            let toFloat b =
                match b with
                | Finite n -> float n
                | Unbounded -> nan   // sentinel; products with nan are filtered
                | SymBound _ -> nan

            let aLo = toFloat ia.lo
            let aHi = toFloat ia.hi
            let bLo = toFloat ib.lo
            let bHi = toFloat ib.hi

            let products =
                [ (aLo, bLo); (aLo, bHi); (aHi, bLo); (aHi, bHi) ]
                |> List.choose (fun (x, y) ->
                    if System.Double.IsNaN x || System.Double.IsNaN y then None
                    elif System.Double.IsInfinity x || System.Double.IsInfinity y then None
                    else Some (x * y))

            if products.IsEmpty then
                Some { lo = Unbounded; hi = Unbounded }
            else
                let minP = int (List.min products)
                let maxP = int (List.max products)
                Some { lo = Finite minP; hi = Finite maxP }


/// intervalNeg: -[a,b] = [-b, -a].
/// Returns None (top) if any bound is symbolic.
let intervalNeg (a: Interval option) : Interval option =
    match a with
    | None -> None
    | Some ia ->
        if not (isConcreteB ia.lo && isConcreteB ia.hi) then None
        else
            let newLo =
                match ia.hi with
                | Unbounded -> Unbounded
                | Finite h  -> Finite (-h)
                | SymBound _ -> Unbounded
            let newHi =
                match ia.lo with
                | Unbounded -> Unbounded
                | Finite l  -> Finite (-l)
                | SymBound _ -> Unbounded
            Some { lo = newLo; hi = newHi }


/// intervalIsExactlyZero: check if interval is definitely exactly zero.
let intervalIsExactlyZero (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.lo, i.hi with
        | Finite 0, Finite 0 -> true
        | _ -> false


/// intervalDefinitelyPositive: check if interval is definitely > 0.
let intervalDefinitelyPositive (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.lo with
        | Finite l -> l > 0
        | _ -> false


/// intervalDefinitelyNonpositive: check if interval is definitely <= 0.
let intervalDefinitelyNonpositive (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.hi with
        | Finite h -> h <= 0
        | _ -> false


/// intervalDefinitelyNegative: check if interval is definitely < 0.
let intervalDefinitelyNegative (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.hi with
        | Finite h -> h < 0
        | _ -> false


/// joinValueRanges: join value_ranges dicts across branches (convex hull per variable).
let joinValueRanges
    (baseline: System.Collections.Generic.Dictionary<string, obj>)
    (branchRanges: System.Collections.Generic.Dictionary<string, obj> list)
    : System.Collections.Generic.Dictionary<string, obj> =

    let allVars = System.Collections.Generic.HashSet<string>()
    for br in branchRanges do
        for kv in br do allVars.Add(kv.Key) |> ignore

    let result = System.Collections.Generic.Dictionary<string, obj>()
    for var in allVars do
        let intervals =
            branchRanges
            |> List.choose (fun br ->
                match br.TryGetValue(var) with
                | true, (:? Interval as iv) -> Some (Some iv)
                | _ -> None)
        match intervals with
        | [] -> ()
        | first :: rest ->
            let joined =
                List.fold (fun acc iv -> joinInterval acc iv) first rest
            match joined with
            | Some iv -> result.[var] <- box iv
            | None -> ()
    result


// ---------------------------------------------------------------------------
// Conditional interval refinement (v1.8.0)
// ---------------------------------------------------------------------------

/// negateComparisonOp: negate a comparison operator for else-branch refinement.
let negateComparisonOp (op: string) : string =
    match op with
    | ">"  -> "<="
    | ">=" -> "<"
    | "<"  -> ">="
    | "<=" -> ">"
    | "==" -> "~="
    | "~=" -> "=="
    | _    -> op


/// intervalFromComparison: convert a comparison to a guard interval.
/// Uses Shapes.addDim for bound arithmetic.
let intervalFromComparison (op: string) (bound: Shapes.Dim) : Interval option =
    match bound with
    | Shapes.Unknown -> None
    | Shapes.Concrete c ->
        match op with
        | ">"  -> Some { lo = Finite (c + 1); hi = Unbounded }
        | ">=" -> Some { lo = Finite c;        hi = Unbounded }
        | "<"  -> Some { lo = Unbounded;       hi = Finite (c - 1) }
        | "<=" -> Some { lo = Unbounded;       hi = Finite c }
        | "==" -> Some { lo = Finite c;        hi = Finite c }
        | "~=" -> None   // not representable as interval
        | _    -> None
    | Shapes.Symbolic s ->
        // Symbolic bound: use SymBound
        match op with
        | ">="  -> Some { lo = SymBound s; hi = Unbounded }
        | "<="  -> Some { lo = Unbounded;  hi = SymBound s }
        | "==" -> Some { lo = SymBound s; hi = SymBound s }
        | _ -> None


/// simpleExprToDim: lightweight dim extractor for use in condition refinement.
/// Handles only Const and Var — does not support BinOp (too complex for here).
let private simpleExprToDim (expr: Ir.Expr) (env: Env.Env) (ctx: Context.AnalysisContext) : Shapes.Dim =
    match expr with
    | Ir.Const(_, _, v) ->
        if v = System.Math.Floor v && not (System.Double.IsInfinity v) then Shapes.Concrete (int v)
        else Shapes.Unknown
    | Ir.Var(_, _, name) ->
        // Check exact interval in value_ranges
        match ctx.cst.valueRanges.TryGetValue(name) with
        | true, (:? Interval as iv) ->
            match iv.lo, iv.hi with
            | Finite lo, Finite hi when lo = hi -> Shapes.Concrete lo
            | _ ->
                match Map.tryFind name env.dimAliases with
                | Some d -> d
                | None   -> Shapes.Symbolic (SymDim.SymDim.var name)
        | _ ->
            match Map.tryFind name env.dimAliases with
            | Some d -> d
            | None   -> Shapes.Symbolic (SymDim.SymDim.var name)
    | _ -> Shapes.Unknown


/// extractConditionRefinements: extract interval refinements from a branch condition.
/// Returns list of (var_name, comparison_op, Dim bound) tuples.
let rec extractConditionRefinements
    (cond: Ir.Expr)
    (env: Env.Env)
    (ctx: Context.AnalysisContext)
    : (string * string * Shapes.Dim) list =

    match cond with
    | Ir.BinOp(_, _, op, left, right) ->
        match op with
        | "&&" | "&" ->
            // Conjunction: merge refinements from both sides
            let lr = extractConditionRefinements left env ctx
            let rr = extractConditionRefinements right env ctx
            lr @ rr
        | "||" | "|" -> []   // disjunction: no safe refinement
        | ">" | ">=" | "<" | "<=" | "==" | "~=" ->
            let getExprBound (e: Ir.Expr) : Shapes.Dim =
                simpleExprToDim e env ctx

            match left, right with
            | Ir.Var(_, _, varName), _ ->
                let bound = getExprBound right
                match bound with
                | Shapes.Unknown -> []
                | _              -> [ (varName, op, bound) ]
            | _, Ir.Var(_, _, varName) ->
                let bound = getExprBound left
                match bound with
                | Shapes.Unknown -> []
                | _ ->
                    // Flip operator: x < 5 is same as 5 > x
                    let flipMap =
                        dict [ ">","<"; ">=","<="; "<",">"; "<=",">="; "==","=="; "~=","~=" ]
                    let flippedOp =
                        match flipMap.TryGetValue(op) with
                        | true, v -> v
                        | _ -> op
                    [ (varName, flippedOp, bound) ]
            | _ -> []
        | _ -> []
    | _ -> []


/// applyRefinements: apply interval refinements to ctx.valueRanges in place.
let applyRefinements
    (ctx: Context.AnalysisContext)
    (refinements: (string * string * Shapes.Dim) list)
    (negate: bool)
    : unit =

    for (varName, op, bound) in refinements do
        let baseIv =
            match ctx.cst.valueRanges.TryGetValue(varName) with
            | true, (:? Interval as iv) -> iv
            | _ -> { lo = Unbounded; hi = Unbounded }

        let actualOp = if negate then negateComparisonOp op else op
        let guardOpt = intervalFromComparison actualOp bound

        match guardOpt with
        | None -> ()
        | Some guard ->
            let refined = meetInterval baseIv guard
            match refined with
            | Some r -> ctx.cst.valueRanges.[varName] <- box r
            | None ->
                // Meet is empty: branch is dead code. Use guard interval to
                // prevent false positives inside unreachable branches.
                ctx.cst.valueRanges.[varName] <- box guard
