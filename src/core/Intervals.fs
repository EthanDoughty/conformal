// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Integer interval abstract domain for scalar value range analysis.
// All operations are constant-time on [lo, hi] pairs. Widening snaps
// to a fixed threshold set (0, 1, 2, 7, 15, 31, 255, 65535) to give
// loops a chance to converge at common values before jumping to
// unbounded.

module Intervals

open SymDim
open SharedTypes

// IntervalBound, Interval types are defined in SharedTypes.fs

let makeInterval (lo: IntervalBound) (hi: IntervalBound) : Interval =
    // Guard: only validate if both concrete ints
    match lo, hi with
    | Finite l, Finite h when l > h ->
        failwith ("Invalid interval: lo=" + string l + " > hi=" + string h)
    | _ -> ()
    { lo = lo; hi = hi }

let intervalUnbounded () : Interval = { lo = Unbounded; hi = Unbounded }

// Check if a bound is a concrete int (not symbolic).
let private isConcreteB (b: IntervalBound) : bool =
    match b with
    | Finite _ | Unbounded -> true
    | SymBound _ -> false


/// Convex hull of two optional intervals (lattice join).
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

/// Push bounds outward to the next threshold when they move,
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


/// Intersection of two intervals (lattice meet).
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


/// [a,b] + [c,d] = [a+c, b+d]. Returns None if any bound is symbolic.
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


/// [a,b] - [c,d] = [a-d, b-c]. Returns None if any bound is symbolic.
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


/// Standard 4-product min/max. Returns None if any bound is symbolic.
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


/// -[a,b] = [-b, -a]. Returns None if any bound is symbolic.
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


/// Check if interval is definitely exactly zero.
let intervalIsExactlyZero (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.lo, i.hi with
        | Finite 0, Finite 0 -> true
        | _ -> false


/// Check if interval is definitely > 0.
let intervalDefinitelyPositive (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.lo with
        | Finite l -> l > 0
        | _ -> false


/// Check if interval is definitely <= 0.
let intervalDefinitelyNonpositive (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.hi with
        | Finite h -> h <= 0
        | _ -> false


/// Check if interval is definitely < 0.
let intervalDefinitelyNegative (iv: Interval option) : bool =
    match iv with
    | None -> false
    | Some i ->
        match i.hi with
        | Finite h -> h < 0
        | _ -> false


/// Join value_ranges maps across branches (convex hull per variable).
/// For each variable in the union of baseline and all branches, collect an interval
/// from each branch (using the baseline as fallback when a branch doesn't mention
/// the variable), then compute the convex hull.
let joinValueRanges
    (baseline: Map<string, Interval>)
    (branchRanges: Map<string, Interval> list)
    : Map<string, Interval> =

    // Collect all variable names from baseline AND all branches.
    let allVars =
        let branchKeys =
            branchRanges
            |> List.collect (fun br -> br |> Map.toList |> List.map fst)
        let baselineKeys = baseline |> Map.toList |> List.map fst
        (branchKeys @ baselineKeys) |> List.distinct

    List.fold (fun acc var ->
        let intervals =
            branchRanges
            |> List.map (fun br ->
                match Map.tryFind var br with
                | Some iv -> Some iv
                | None    ->
                    // Branch didn't modify this variable: use baseline as fallback
                    Map.tryFind var baseline)
        // Filter to only branches that produced an interval
        let present = intervals |> List.choose (fun iv -> match iv with Some i -> Some (Some i) | None -> None)
        match present with
        | [] -> acc
        | first :: rest ->
            let joined = List.fold joinInterval first rest
            match joined with
            | Some iv -> Map.add var iv acc
            | None    -> acc
    ) Map.empty allVars


/// Widen intervals that moved between baseline and current state.
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


/// Tighten a widened bound using the new iterate from the narrowing pass.
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


/// Apply one narrowing pass to all intervals.
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

/// Negate a comparison operator for else-branch refinement.
let negateComparisonOp (op: string) : string =
    match op with
    | ">"  -> "<="
    | ">=" -> "<"
    | "<"  -> ">="
    | "<=" -> ">"
    | "==" -> "~="
    | "~=" -> "=="
    | _    -> op


/// Convert a comparison to a guard interval.
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


// Lightweight dim extractor for condition refinement. Handles Const and Var only.
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
    | _ -> Shapes.Unknown  // open type: all other Expr cases not handled here


/// Extract interval refinements from a branch condition.
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


/// When an interval is exact [k,k], propagate k into DimEquiv and
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


/// Apply interval refinements to ctx.valueRanges in place.
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
        | None ->
            // Special case: ~= with concrete bound at endpoint of current interval
            if actualOp = "~=" then
                match bound with
                | Shapes.Concrete c ->
                    match baseIv.lo, baseIv.hi with
                    | Finite lo, Finite hi when lo = c && hi = c ->
                        // var in [c,c] and var ~= c: dead branch — use guard to prevent FPs
                        ctx.cst.valueRanges <- Map.add varName { lo = Finite c; hi = Finite c } ctx.cst.valueRanges
                    | Finite lo, _ when lo = c ->
                        let r = { baseIv with lo = Finite (c + 1) }
                        ctx.cst.valueRanges <- Map.add varName r ctx.cst.valueRanges
                        bridgeToDimEquiv ctx varName r
                    | _, Finite hi when hi = c ->
                        let r = { baseIv with hi = Finite (c - 1) }
                        ctx.cst.valueRanges <- Map.add varName r ctx.cst.valueRanges
                        bridgeToDimEquiv ctx varName r
                    | _ -> ()  // interior exclusion: can't represent as single interval
                | _ -> ()
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

/// Intersect two upper-bound maps (keep only facts present in both).
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


/// Same as joinUpperBounds (intersection).
/// Since the set of bounds only shrinks, this naturally converges.
let widenUpperBounds
    (oldMap: Map<string, string * int>)
    (newMap: Map<string, string * int>)
    : Map<string, string * int> =
    joinUpperBounds oldMap newMap


/// Remove all bounds that mention varName (either as the constrained variable or as
/// the bound variable) after an assignment to varName.
let killUpperBoundsFor (varName: string) (bounds: Map<string, string * int>) : Map<string, string * int> =
    bounds
    |> Map.filter (fun key (bVar, _) -> key <> varName && bVar <> varName)


/// Decompose an expression into (baseVar, offset) if it is Var or Var +/- Const.
/// Returns None for anything more complex (e.g. 2*i, i+j, etc.).
let tryDecomposeVarPlusConst (expr: Ir.Expr) : (string * int) option =
    match expr with
    | Ir.Var(_, name) -> Some (name, 0)
    | Ir.BinOp(_, "+", Ir.Var(_, name), Ir.Const(_, v))
    | Ir.BinOp(_, "+", Ir.Const(_, v), Ir.Var(_, name))
        when v = System.Math.Floor(v) -> Some (name, int v)
    | Ir.BinOp(_, "-", Ir.Var(_, name), Ir.Const(_, v))
        when v = System.Math.Floor(v) -> Some (name, -(int v))
    | _ -> None


/// Check if the Pentagon upper-bound relation proves that the expression
/// (varName + exprOffset) is within matrix dimension matDim. Returns true if:
///   (a) upperBounds[varName] = (boundVar, pentOffset) and boundVar has exact
///       interval [k,k] in valueRanges and k + pentOffset + exprOffset <= concrete matDim size, OR
///   (b) upperBounds[varName] = (boundVar, pentOffset) and the DimEquiv root of
///       boundVar matches matDim's symbolic key and pentOffset + exprOffset <= 0.
let pentagonProvesInBoundsWithOffset
    (ctx: Context.AnalysisContext)
    (varName: string)
    (exprOffset: int)
    (matDim: Shapes.Dim)
    : bool =
    match Map.tryFind varName ctx.cst.upperBounds with
    | None -> false
    | Some (boundVar, pentOffset) ->
        let totalOffset = pentOffset + exprOffset
        // Case (a): boundVar is concretely known and k + totalOffset <= concrete matDim
        let concreteProof =
            match Map.tryFind boundVar ctx.cst.valueRanges with
            | Some iv ->
                match iv.lo, iv.hi with
                | Finite lo, Finite hi when lo = hi ->
                    match matDim with
                    | Shapes.Concrete ms -> lo + totalOffset <= ms
                    | _ -> false
                | _ -> false
            | None -> false
        if concreteProof then true
        else
        // Case (b): boundVar's DimEquiv root matches matDim's symbolic key and totalOffset <= 0
        match matDim with
        | Shapes.Symbolic _ when totalOffset <= 0 ->
            let matDimKey = Shapes.dimStr matDim
            let boundVarRoot = DimEquiv.find ctx.cst.dimEquiv boundVar
            let matDimRoot = DimEquiv.find ctx.cst.dimEquiv matDimKey
            boundVarRoot = matDimRoot
        | _ -> false


/// Check if the Pentagon upper-bound relation proves that index variable varName is
/// within matrix dimension matDim (no expression offset).
let pentagonProvesInBounds
    (ctx: Context.AnalysisContext)
    (varName: string)
    (matDim: Shapes.Dim)
    : bool =
    pentagonProvesInBoundsWithOffset ctx varName 0 matDim


/// Intersect two lower-bound maps (keep only facts present in both).
/// Symmetric with joinUpperBounds -- intersection is the sound join for relational facts.
let joinLowerBounds
    (a: Map<string, string * int>)
    (b: Map<string, string * int>)
    : Map<string, string * int> =
    a |> Map.filter (fun varName boundA ->
        match Map.tryFind varName b with
        | Some boundB -> boundA = boundB
        | None -> false)


/// Same as joinLowerBounds (intersection converges trivially).
let widenLowerBounds
    (oldMap: Map<string, string * int>)
    (newMap: Map<string, string * int>)
    : Map<string, string * int> =
    joinLowerBounds oldMap newMap


/// Remove all lower bounds that mention varName after an assignment.
let killLowerBoundsFor (varName: string) (bounds: Map<string, string * int>) : Map<string, string * int> =
    bounds
    |> Map.filter (fun key (bVar, _) -> key <> varName && bVar <> varName)


/// Check if the Pentagon lower-bound relation proves that (varName + exprOffset) >= 1.
/// Returns true when:
///   (a) lowerBounds[varName] = (boundVar, pentOffset) and boundVar has exact
///       interval [k,k] and k + pentOffset + exprOffset >= 1, OR
///   (b) lowerBounds[varName] = (boundVar, pentOffset) and boundVar has a symbolic
///       DimEquiv root and pentOffset + exprOffset >= 0 (i.e. var+offset >= boundVar >= 1
///       by convention that symbolic dims represent valid sizes >= 1).
let pentagonProvesLowerBoundWithOffset
    (ctx: Context.AnalysisContext)
    (varName: string)
    (exprOffset: int)
    : bool =
    match Map.tryFind varName ctx.cst.lowerBounds with
    | None -> false
    | Some (boundVar, pentOffset) ->
        let totalOffset = pentOffset + exprOffset
        // Case (a): boundVar is concretely known
        match Map.tryFind boundVar ctx.cst.valueRanges with
        | Some iv ->
            match iv.lo, iv.hi with
            | Finite lo, Finite hi when lo = hi -> lo + totalOffset >= 1
            | _ -> false
        | None ->
            // Case (b): symbolic bound -- if the lower bound is a symbolic var and
            // totalOffset >= 0, then varName + exprOffset >= boundVar + pentOffset + exprOffset
            // >= 1 + 0 = 1 (symbolic dims are >= 1).
            totalOffset >= 0


/// Check if the Pentagon lower-bound relation proves that index variable varName is >= 1.
/// Returns true when lowerBounds[varName] = (boundVar, offset) and the bound implies >= 1.
let pentagonProvesLowerBound
    (ctx: Context.AnalysisContext)
    (varName: string)
    : bool =
    pentagonProvesLowerBoundWithOffset ctx varName 0


/// Parse a while-condition expression and extract relational bounds of the form
/// (varName, boundVar, offset, isUpper). Only fires when both sides are simple Var
/// references; constant bounds are already handled by extractConditionRefinements.
///
/// Patterns recognised:
///   i <= n  ->  (i, n, 0, true)   -- upper bound: i <= n + 0
///   i <  n  ->  (i, n, -1, true)  -- upper bound: i <= n + (-1)
///   i >= n  ->  (i, n, 0, false)  -- lower bound: i >= n + 0
///   i >  n  ->  (i, n, 1, false)  -- lower bound: i >= n + 1
///   cond1 && cond2 -> merge both sides
let rec extractPentagonBoundsFromCondition
    (cond: Ir.Expr)
    : (string * string * int * bool) list =
    match cond with
    | Ir.BinOp(_, "&&", left, right) | Ir.BinOp(_, "&", left, right) ->
        extractPentagonBoundsFromCondition left @
        extractPentagonBoundsFromCondition right
    | Ir.BinOp(_, "<=", Ir.Var(_, varName), Ir.Var(_, boundVar)) ->
        [ (varName, boundVar, 0, true) ]
    | Ir.BinOp(_, "<", Ir.Var(_, varName), Ir.Var(_, boundVar)) ->
        [ (varName, boundVar, -1, true) ]
    | Ir.BinOp(_, ">=", Ir.Var(_, varName), Ir.Var(_, boundVar)) ->
        [ (varName, boundVar, 0, false) ]
    | Ir.BinOp(_, ">", Ir.Var(_, varName), Ir.Var(_, boundVar)) ->
        [ (varName, boundVar, 1, false) ]
    // Q2: Var +/- Const on either side of a comparison.
    // e.g. `i+1 <= n`  -> i <= n + (-1)  i.e. (varName="i", boundVar="n", offset=-1, isUpper=true)
    //      `i <= n-1`  -> i <= n + (-1)
    //      `i-1 >= n`  -> i >= n + 1
    | Ir.BinOp(_, "<=", lhs, rhs) ->
        match tryDecomposeVarPlusConst lhs, tryDecomposeVarPlusConst rhs with
        | Some (varName, lhsOff), Some (boundVar, rhsOff) when varName <> boundVar ->
            // varName + lhsOff <= boundVar + rhsOff  ->  varName <= boundVar + (rhsOff - lhsOff)
            [ (varName, boundVar, rhsOff - lhsOff, true) ]
        | _ -> []
    | Ir.BinOp(_, "<", lhs, rhs) ->
        match tryDecomposeVarPlusConst lhs, tryDecomposeVarPlusConst rhs with
        | Some (varName, lhsOff), Some (boundVar, rhsOff) when varName <> boundVar ->
            [ (varName, boundVar, rhsOff - lhsOff - 1, true) ]
        | _ -> []
    | Ir.BinOp(_, ">=", lhs, rhs) ->
        match tryDecomposeVarPlusConst lhs, tryDecomposeVarPlusConst rhs with
        | Some (varName, lhsOff), Some (boundVar, rhsOff) when varName <> boundVar ->
            // varName + lhsOff >= boundVar + rhsOff  ->  varName >= boundVar + (rhsOff - lhsOff)
            [ (varName, boundVar, rhsOff - lhsOff, false) ]
        | _ -> []
    | Ir.BinOp(_, ">", lhs, rhs) ->
        match tryDecomposeVarPlusConst lhs, tryDecomposeVarPlusConst rhs with
        | Some (varName, lhsOff), Some (boundVar, rhsOff) when varName <> boundVar ->
            [ (varName, boundVar, rhsOff - lhsOff + 1, false) ]
        | _ -> []
    // Note: flipped patterns (n >= i → i <= n) are structurally identical to
    // the above in F# pattern matching, so the left Var is always treated as
    // the variable and the right as the bound. This is correct for the common
    // MATLAB idiom `while i <= n`.
    | _ -> []


/// For each x in upperBounds where upperBounds[x] = (y, c)
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


/// For each x in lowerBounds where lowerBounds[x] = (y, c)
/// and valueRanges[y] = [k,k] (exact), tighten valueRanges[x].lo to max(current_lo, k+c).
let applyPentagonLowerBridge (ctx: Context.AnalysisContext) : unit =
    for kvp in ctx.cst.lowerBounds |> Map.toSeq do
        let (varName, (boundVar, offset)) = kvp
        match Map.tryFind boundVar ctx.cst.valueRanges with
        | Some iv ->
            match iv.lo, iv.hi with
            | Finite lo, Finite hi when lo = hi ->
                // boundVar is exactly known: tighten varName's lower bound
                let tightenedLo = Finite (lo + offset)
                let existing =
                    match Map.tryFind varName ctx.cst.valueRanges with
                    | Some existing -> existing
                    | None -> { lo = Unbounded; hi = Unbounded }
                let newLo =
                    match existing.lo with
                    | Finite l -> Finite (max l (lo + offset))
                    | Unbounded -> tightenedLo
                    | SymBound _ -> tightenedLo
                let newIv = { existing with lo = newLo }
                if newIv <> existing then
                    ctx.cst.valueRanges <- Map.add varName newIv ctx.cst.valueRanges
            | _ -> ()
        | None -> ()
