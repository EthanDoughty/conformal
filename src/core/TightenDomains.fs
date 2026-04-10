// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Formal reduced product across the interval, Pentagon, and symbolic
// dimension domains. Combines information from each domain so that a
// fact discovered by one can refine the others.
//
// Earlier versions of the analyzer handled the interaction between
// domains with inline bridge calls scattered throughout the pipeline,
// and the order they ran in mattered, which often meant late facts
// could not flow back to refine earlier results. The reduced product
// is the fix. If intervals pin a variable to an exact value, DimEquiv
// can accept that value as concrete, which may unlock dimension
// constraints like "this matrix has that many rows". That new fact
// can then refine other intervals that reference the same variable,
// and so on.
//
// The implementation runs in a sequence of bounded phases.
// Bidirectional Interval to DimEquiv propagation runs first, capped
// at a fixed iteration limit. A Pentagon to Interval pass follows.
// Finally, a shape resolution using the updated dimension store. The
// iteration cap is what guarantees termination without needing a
// convergence check.

module TightenDomains

open SharedTypes
open Context
open Intervals
open DimEquiv
open Constraints

// Phase 1: propagate exact intervals [k,k] into DimEquiv and back.
// Returns true if any change occurred in this iteration.
let private propagateConcretes (ctx: AnalysisContext) : bool =
    let mutable changed = false

    // Forward: Interval [k,k] -> DimEquiv.setConcrete
    for kvp in ctx.cst.valueRanges |> Map.toSeq do
        let (varName, iv) = kvp
        match iv.lo, iv.hi with
        | Finite lo, Finite hi when lo = hi ->
            let wasNew = DimEquiv.setConcrete ctx.cst.dimEquiv varName lo
            if wasNew then changed <- true
        | _ -> ()

    // Backward: DimEquiv concrete -> valueRanges for simple variable keys
    for key in ctx.cst.dimEquiv.parent.Keys |> Seq.toArray do
        let isSimpleVar =
            key.Length > 0 &&
            key |> Seq.forall (fun c -> System.Char.IsLetterOrDigit c || c = '_')
        if isSimpleVar then
            match DimEquiv.getConcrete ctx.cst.dimEquiv key with
            | Some n ->
                let exact = { lo = Finite n; hi = Finite n }
                match Map.tryFind key ctx.cst.valueRanges with
                | Some existing when existing = exact -> ()  // already correct
                | Some _ -> ()  // key has a current value; don't overwrite with stale DimEquiv
                | None ->
                    ctx.cst.valueRanges <- Map.add key exact ctx.cst.valueRanges
                    changed <- true
            | None -> ()

    changed


// Phase 2: Pentagon -> Interval tightening, single pass.
let private applyPentagonBridges (ctx: AnalysisContext) : unit =
    Intervals.applyPentagonBridge ctx
    Intervals.applyPentagonLowerBridge ctx


// Phase 3: Resolve symbolic dims in env shapes using current knowledge.
let private resolveShapes (ctx: AnalysisContext) (env: Env.Env) : unit =
    for kvp in env.bindings |> Map.toSeq do
        let (varName, shape) = kvp
        let resolved = Constraints.resolveShape ctx shape
        if resolved <> shape then
            Env.Env.set env varName resolved


/// Tighten all domains using mutual information.
/// Phase 1: Propagate concretes (Interval <-> DimEquiv), iterate until stable (max 3)
/// Phase 2: Apply Pentagon bridges (Pentagon -> Interval), single pass
/// Phase 3: Resolve symbolic dims in shapes using current knowledge
let tightenDomains (ctx: AnalysisContext) (env: Env.Env) : unit =
    // Phase 1: iterate until stable, capped at 3 iterations
    let mutable i = 0
    let mutable anyChanged = false
    let mutable keepGoing = true
    while keepGoing && i < 3 do
        let changed = propagateConcretes ctx
        if changed then anyChanged <- true
        i <- i + 1
        if not changed then keepGoing <- false

    // Phase 2: Pentagon -> Interval (single pass, non-recursive)
    applyPentagonBridges ctx

    // Phase 3: eager shape resolution (skip if no concretes propagated)
    if anyChanged then
        resolveShapes ctx env
