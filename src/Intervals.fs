module Intervals

open SymDim
open SharedTypes

// ---------------------------------------------------------------------------
// Integer interval abstract domain for value range analysis.
// Port of analysis/intervals.py
// ---------------------------------------------------------------------------

// IntervalBound, Interval types are defined in SharedTypes.fs

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


/// Threshold set for widening (sorted ascending).
/// When a bound moves outward, it is snapped to the next threshold in the
/// direction of movement.  This keeps intervals finite longer and gives the
/// fixpoint loop more iterations at useful precision before collapsing to
/// Unbounded.  Convergence is still guaranteed: each widening step moves the
/// bound to a strictly larger threshold or to Unbounded; the set is finite.
let private WIDEN_THRESHOLDS = [| -1000; -100; -10; -1; 0; 1; 10; 100; 1000 |]

/// widenInterval: push bounds outward to the next threshold when they move,
/// falling back to Unbounded only when no threshold covers the new value.
let widenInterval (oldIv: Interval) (newIv: Interval) : Interval =
    // Lower bound: when it decreases, snap to the next threshold <= nl.
    let widenLo oldBound newBound =
        match oldBound, newBound with
        | Unbounded, _ -> Unbounded        // already widened to -inf, preserve
        | _, Unbounded -> Unbounded        // body produced unknown lower bound; must not narrow
        | Finite ol, Finite nl ->
            if nl < ol then
                // Lower bound decreased: snap to largest threshold <= nl
                match WIDEN_THRESHOLDS |> Array.tryFindBack (fun t -> t <= nl) with
                | Some thresh -> Finite thresh
                | None        -> Unbounded  // below all thresholds
            else Finite ol  // stable or increased: keep old bound
        | _ -> Unbounded   // symbolic: can't compare, widen to -inf (sound)

    // Upper bound: when it increases, snap to the next threshold >= nh.
    let widenHi oldBound newBound =
        match oldBound, newBound with
        | Unbounded, _ -> Unbounded        // already widened to +inf, preserve
        | _, Unbounded -> Unbounded        // body produced unknown upper bound; must not narrow
        | Finite oh, Finite nh ->
            if nh > oh then
                // Upper bound increased: snap to smallest threshold >= nh
                match WIDEN_THRESHOLDS |> Array.tryFind (fun t -> t >= nh) with
                | Some thresh -> Finite thresh
                | None        -> Unbounded  // above all thresholds
            else Finite oh  // stable or decreased: keep old bound
        | _ -> Unbounded   // symbolic: can't compare, widen to +inf (sound)

    { lo = widenLo oldIv.lo newIv.lo; hi = widenHi oldIv.hi newIv.hi }


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


/// joinValueRanges: join value_ranges maps across branches (convex hull per variable).
let joinValueRanges
    (baseline: Map<string, Interval>)
    (branchRanges: Map<string, Interval> list)
    : Map<string, Interval> =

    let allVars =
        branchRanges
        |> List.collect (fun br -> br |> Map.toList |> List.map fst)
        |> List.distinct

    List.fold (fun acc var ->
        let intervals =
            branchRanges
            |> List.choose (fun br ->
                match Map.tryFind var br with
                | Some iv -> Some (Some iv)
                | None    -> None)
        match intervals with
        | [] -> acc
        | first :: rest ->
            let joined = List.fold joinInterval first rest
            match joined with
            | Some iv -> Map.add var iv acc
            | None    -> acc
    ) Map.empty allVars


/// widenValueRanges: widen intervals that moved between baseline and current state.
/// Variables whose interval did not change are preserved.
/// Variables new in current (not in baseline) are kept as-is.
/// The loop iteration variable (if any) is excluded from widening.
/// Returns the widened map (does not mutate).
let widenValueRanges
    (baseline: Map<string, Interval>)
    (current: Map<string, Interval>)
    (exclude: string option)
    : Map<string, Interval> =
    current |> Map.map (fun key iv ->
        if Some key = exclude then iv
        else
            match Map.tryFind key baseline with
            | Some oldIv when oldIv <> iv -> widenInterval oldIv iv
            | _ -> iv)  // new variable or unchanged: keep as-is


/// narrowInterval: tighten a widened bound using the new iterate from the narrowing pass.
/// Only moves bounds inward (more precise); never widens.
let narrowInterval (wideIv: Interval) (newIv: Interval) : Interval =
    // Lower bound: take max (move floor up if iterate has higher lower bound)
    let narrowLo wideBound newBound =
        match wideBound, newBound with
        | Unbounded, other -> other           // -inf narrows to anything tighter
        | _, Unbounded -> wideBound           // new iterate is -inf: no tightening
        | Finite wl, Finite nl -> Finite (max wl nl)  // take the tighter floor
        | _ -> wideBound                      // can't compare: keep wide (sound)
    // Upper bound: take min (move ceiling down if iterate has lower upper bound)
    let narrowHi wideBound newBound =
        match wideBound, newBound with
        | Unbounded, other -> other           // +inf narrows to anything tighter
        | _, Unbounded -> wideBound           // new iterate is +inf: no tightening
        | Finite wh, Finite nh -> Finite (min wh nh)  // take the tighter ceiling
        | _ -> wideBound                      // can't compare: keep wide (sound)
    { lo = narrowLo wideIv.lo newIv.lo; hi = narrowHi wideIv.hi newIv.hi }


/// narrowValueRanges: apply one narrowing pass to all intervals.
/// For each variable in the widened map, if the narrowing iterate has a tighter
/// bound, adopt it.  Variables that disappeared in the iterate are kept wide (sound).
/// The loop iteration variable (if any) is excluded from narrowing.
let narrowValueRanges
    (widened: Map<string, Interval>)
    (narrowed: Map<string, Interval>)
    (exclude: string option)
    : Map<string, Interval> =
    widened |> Map.map (fun key iv ->
        if Some key = exclude then iv
        else
            match Map.tryFind key narrowed with
            | Some newIv -> narrowInterval iv newIv
            | None -> iv)  // variable disappeared: keep widened (sound)


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
    | Shapes.Unknown  -> None
    | Shapes.Range _  -> None   // Range bound: too imprecise to refine intervals
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
    | Ir.Const(_, v) ->
        if v = System.Math.Floor(v : float) && not (System.Double.IsInfinity v) then Shapes.Concrete (int v)
        else Shapes.Unknown
    | Ir.Var(_, name) ->
        // Check exact interval in value_ranges
        match Map.tryFind name ctx.cst.valueRanges with
        | Some iv ->
            match iv.lo, iv.hi with
            | Finite lo, Finite hi when lo = hi -> Shapes.Concrete lo
            | _ ->
                match Map.tryFind name env.dimAliases with
                | Some d -> d
                | None   -> Shapes.Symbolic (SymDim.SymDim.var name)
        | None ->
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
    | Ir.BinOp(_, op, left, right) ->
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
            | Ir.Var(_, varName), _ ->
                let bound = getExprBound right
                match bound with
                | Shapes.Unknown -> []
                | _              -> [ (varName, op, bound) ]
            | _, Ir.Var(_, varName) ->
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


/// bridgeToDimEquiv: when an interval is exact [k,k], propagate k into DimEquiv and
/// back-propagate the concrete value to valueRanges for all equivalent simple variables.
let bridgeToDimEquiv (ctx: Context.AnalysisContext) (varName: string) (iv: Interval) : unit =
    match iv.lo, iv.hi with
    | Finite lo, Finite hi when lo = hi ->
        DimEquiv.setConcrete ctx.cst.dimEquiv varName lo |> ignore
        // Back-propagate: update valueRanges for all simple-variable keys in the SAME class.
        let root = DimEquiv.find ctx.cst.dimEquiv varName
        let exact = { lo = Finite lo; hi = Finite hi }
        for key in ctx.cst.dimEquiv.parent.Keys |> Seq.toArray do
            // Only propagate to simple identifier-style keys (no operators/spaces).
            let isSimpleVar = key |> Seq.forall (fun c -> System.Char.IsLetterOrDigit c || c = '_')
            if isSimpleVar && DimEquiv.find ctx.cst.dimEquiv key = root then
                ctx.cst.valueRanges <- Map.add key exact ctx.cst.valueRanges
    | _ -> ()


/// applyRefinements: apply interval refinements to ctx.valueRanges in place.
let applyRefinements
    (ctx: Context.AnalysisContext)
    (refinements: (string * string * Shapes.Dim) list)
    (negate: bool)
    : unit =

    for (varName, op, bound) in refinements do
        let baseIv =
            match Map.tryFind varName ctx.cst.valueRanges with
            | Some iv -> iv
            | None    -> { lo = Unbounded; hi = Unbounded }

        let actualOp = if negate then negateComparisonOp op else op
        let guardOpt = intervalFromComparison actualOp bound

        match guardOpt with
        | None -> ()
        | Some guard ->
            let refined = meetInterval baseIv guard
            match refined with
            | Some r ->
                ctx.cst.valueRanges <- Map.add varName r ctx.cst.valueRanges
                bridgeToDimEquiv ctx varName r
            | None ->
                // Meet is empty: branch is dead code. Use guard interval to
                // prevent false positives inside unreachable branches.
                ctx.cst.valueRanges <- Map.add varName guard ctx.cst.valueRanges
                bridgeToDimEquiv ctx varName guard


// ---------------------------------------------------------------------------
// Pentagon domain helpers (relational upper-bound constraints)
// ---------------------------------------------------------------------------

/// joinUpperBounds: intersect two upper-bound maps (keep only facts present in both).
/// When the same variable appears in both maps with different offsets, take the max offset
/// (least precise but sound: a larger offset is a weaker constraint).
let joinUpperBounds
    (a: Map<string, string * int>)
    (b: Map<string, string * int>)
    : Map<string, string * int> =
    // Keep only variables bound in BOTH maps.
    a |> Map.filter (fun varName boundA ->
        match Map.tryFind varName b with
        | Some boundB -> boundA = boundB   // keep only when the bound is identical
        | None -> false)


/// widenUpperBounds: same as joinUpperBounds (intersection).
/// Since the set of bounds only shrinks, this naturally converges.
let widenUpperBounds
    (oldMap: Map<string, string * int>)
    (newMap: Map<string, string * int>)
    : Map<string, string * int> =
    joinUpperBounds oldMap newMap


/// killUpperBoundsFor: remove all bounds that mention varName (either as the
/// constrained variable or as the bound variable) after an assignment to varName.
let killUpperBoundsFor (varName: string) (bounds: Map<string, string * int>) : Map<string, string * int> =
    bounds
    |> Map.filter (fun key (bVar, _) -> key <> varName && bVar <> varName)


/// applyPentagonBridge: for each x in upperBounds where upperBounds[x] = (y, c)
/// and valueRanges[y] = [k,k] (exact), tighten valueRanges[x].hi to min(current_hi, k+c).
let applyPentagonBridge (ctx: Context.AnalysisContext) : unit =
    for kvp in ctx.cst.upperBounds |> Map.toSeq do
        let (varName, (boundVar, offset)) = kvp
        match Map.tryFind boundVar ctx.cst.valueRanges with
        | Some iv ->
            match iv.lo, iv.hi with
            | Finite lo, Finite hi when lo = hi ->
                // boundVar is exactly known: tighten varName's upper bound
                let tightenedHi = Finite (hi + offset)
                let existing =
                    match Map.tryFind varName ctx.cst.valueRanges with
                    | Some existing -> existing
                    | None -> { lo = Unbounded; hi = Unbounded }
                let newHi =
                    match existing.hi with
                    | Finite h -> Finite (min h (hi + offset))
                    | Unbounded -> tightenedHi
                    | SymBound _ -> tightenedHi
                let newIv = { existing with hi = newHi }
                if newIv <> existing then
                    ctx.cst.valueRanges <- Map.add varName newIv ctx.cst.valueRanges
            | _ -> ()
        | None -> ()
