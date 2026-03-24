module TightenDomains

open SharedTypes
open Context
open Intervals
open DimEquiv
open Constraints

// ---------------------------------------------------------------------------
// Formal reduced-product bridge: tightens all abstract domains using mutual
// information.  Called at well-defined program points (after refinements, at
// loop body entry, after scalar assignment) rather than scattered inline.
//
// Phase 1: Interval <-> DimEquiv concrete propagation, iterate until stable
//          (max 3 iterations).
// Phase 2: Pentagon -> Interval tightening (single pass).
// Phase 3: Resolve symbolic dims in shapes using current knowledge.
// ---------------------------------------------------------------------------

/// Phase 1: propagate exact intervals [k,k] into DimEquiv and back.
/// Returns true if any change occurred in this iteration.
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


/// Phase 2: Pentagon -> Interval tightening, single pass.
let private applyPentagonBridges (ctx: AnalysisContext) : unit =
    Intervals.applyPentagonBridge ctx
    Intervals.applyPentagonLowerBridge ctx


/// Phase 3: Resolve symbolic dims in env shapes using current knowledge.
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
    let mutable keepGoing = true
    while keepGoing && i < 3 do
        let changed = propagateConcretes ctx
        i <- i + 1
        if not changed then keepGoing <- false

    // Phase 2: Pentagon -> Interval (single pass, non-recursive)
    applyPentagonBridges ctx

    // Phase 3: eager shape resolution
    resolveShapes ctx env
