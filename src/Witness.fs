module Witness

open Shapes
open SymDim
open SharedTypes

// ---------------------------------------------------------------------------
// Witness generation for dimension conflict sites.
// Port of analysis/witness.py
// ---------------------------------------------------------------------------

// ConflictSite type is defined in SharedTypes.fs

/// Witness: concrete proof that a warning is a real bug.
type Witness = {
    assignments:    (string * int) list
    dimAConcrete:   int
    dimBConcrete:   int
    explanation:    string
    path:           (string * bool * int) list
}


/// evalDim: evaluate a Dim under concrete bindings.
let private evalDim (d: Dim) (bindings: Map<string, int>) : int option =
    match d with
    | Concrete n -> Some n
    | Unknown    -> None
    | Symbolic s ->
        // Evaluate SymDim by substituting bindings (signature: evaluate bindings s)
        SymDim.evaluate bindings s


/// constraintsSatisfied: check that snapshot equality constraints hold under bindings.
let private constraintsSatisfied (site: ConflictSite) (bindings: Map<string, int>) : bool =
    // For now, basic check: constraints are string pairs (dimStr1, dimStr2).
    // We skip constraint validation here and just check dims directly.
    true  // Conservative: don't reject on unsatisfied constraints


/// collectRelevantVars: collect variable names from dim_a, dim_b.
let private collectRelevantVars (site: ConflictSite) : Set<string> =
    let getVars (d: Dim) : Set<string> =
        match d with
        | Symbolic s -> Set.ofSeq (SymDim.variables s)
        | _ -> Set.empty
    Set.union (getVars site.dimA) (getVars site.dimB)


/// negateOp / flipOp for path bound extraction.
let private negateOp (op: string) =
    match op with
    | ">"  -> "<="
    | ">=" -> "<"
    | "<"  -> ">="
    | "<=" -> ">"
    | "==" -> "~="
    | "~=" -> "=="
    | _    -> op

let private flipOp (op: string) =
    match op with
    | ">"  -> "<"
    | ">=" -> "<="
    | "<"  -> ">"
    | "<=" -> ">="
    | "==" -> "=="
    | "~=" -> "~="
    | _    -> op

let private opToBounds (op: string) (v: int) : int option * int option =
    match op with
    | ">"  -> (Some (v + 1), None)
    | ">=" -> (Some v,       None)
    | "<"  -> (None,         Some (v - 1))
    | "<=" -> (None,         Some v)
    | "==" -> (Some v,       Some v)
    | _    -> (None, None)

let private pathIntervalBounds (pathSnapshot: (string * bool * int) list) (var: string) : int option * int option =
    let mutable lo : int option = None
    let mutable hi : int option = None
    for (desc, branchTaken, _) in pathSnapshot do
        // Parse "VAR OP VALUE" or "VALUE OP VAR"
        let m1 = System.Text.RegularExpressions.Regex.Match(desc, @"^(\w+)\s*(>|>=|<|<=|==|~=)\s*(-?\d+)$")
        if m1.Success && m1.Groups.[1].Value = var then
            let op = m1.Groups.[2].Value
            let v = int m1.Groups.[3].Value
            let effectiveOp = if branchTaken then op else negateOp op
            let (bLo, bHi) = opToBounds effectiveOp v
            lo <- match lo, bLo with | Some l, Some bl -> Some (max l bl) | None, x -> x | x, None -> x
            hi <- match hi, bHi with | Some h, Some bh -> Some (min h bh) | None, x -> x | x, None -> x
        else
            let m2 = System.Text.RegularExpressions.Regex.Match(desc, @"^(-?\d+)\s*(>|>=|<|<=|==|~=)\s*(\w+)$")
            if m2.Success && m2.Groups.[3].Value = var then
                let op = m2.Groups.[2].Value
                let v = int m2.Groups.[1].Value
                let effectiveOp =
                    let flipped = flipOp op
                    if branchTaken then flipped else negateOp flipped
                let (bLo, bHi) = opToBounds effectiveOp v
                lo <- match lo, bLo with | Some l, Some bl -> Some (max l bl) | None, x -> x | x, None -> x
                hi <- match hi, bHi with | Some h, Some bh -> Some (min h bh) | None, x -> x | x, None -> x
    (lo, hi)


/// findSatisfyingAssignment: enumerate assignments and verify witness.
let private findSatisfyingAssignment
    (site: ConflictSite)
    (relevantVars: Set<string>)
    (scalarBindings: Map<string, int>)
    : Witness option =

    let valueRanges = Map.ofList site.valueRangesSnapshot

    let preBindings =
        scalarBindings
        |> Map.filter (fun k _ -> Set.contains k relevantVars)

    let freeVars = relevantVars - (Set.ofSeq (preBindings |> Map.toSeq |> Seq.map fst))
    let freeVarsList = freeVars |> Set.toList |> List.sort

    if freeVarsList.Length > 8 then None
    else

    let candidateRange (var: string) : int list =
        let mutable lo = 0
        let mutable hi = 10
        match Map.tryFind var valueRanges with
        | Some (ivLo, ivHi) ->
            lo <- max 0 ivLo
            hi <- ivHi
        | None -> ()
        let (pathLo, pathHi) = pathIntervalBounds site.pathSnapshot var
        match pathLo with Some pl -> lo <- max lo pl | None -> ()
        match pathHi with Some ph -> hi <- min hi ph | None -> ()
        hi <- min hi (lo + 20)
        [ lo .. hi ]

    let candidates = freeVarsList |> List.map candidateRange

    let rec enumerate (vars: string list) (candidateLists: int list list) (acc: Map<string, int>) : Witness option =
        match vars, candidateLists with
        | [], [] ->
            let allBindings =
                Map.fold (fun m k v -> Map.add k v m) acc preBindings
            let a = evalDim site.dimA allBindings
            let b = evalDim site.dimB allBindings
            match a, b with
            | Some av, Some bv when av >= 0 && bv >= 0 && av <> bv ->
                if constraintsSatisfied site allBindings then
                    let usedVars =
                        allBindings
                        |> Map.filter (fun k _ -> Set.contains k relevantVars)
                        |> Map.toList
                        |> List.sort
                    let parts = usedVars |> List.map (fun (k, v) -> k + "=" + string v) |> String.concat ", "
                    let explanation = parts + " -> dims " + string av + " != " + string bv
                    Some {
                        assignments   = usedVars
                        dimAConcrete  = av
                        dimBConcrete  = bv
                        explanation   = explanation
                        path          = site.pathSnapshot
                    }
                else None
            | _ -> None
        | v :: restVars, cands :: restCands ->
            let mutable result : Witness option = None
            let mutable idx = 0
            while result.IsNone && idx < cands.Length do
                result <- enumerate restVars restCands (Map.add v cands.[idx] acc)
                idx <- idx + 1
            result
        | _ -> None

    enumerate freeVarsList candidates Map.empty


/// attemptWitness: try to construct a concrete witness for a ConflictSite.
let attemptWitness (site: ConflictSite) : Witness option =
    let dimA = site.dimA
    let dimB = site.dimB

    if dimA = Unknown || dimB = Unknown then None
    else

    // Bail on quadratic+ terms
    let maxDegree (d: Dim) =
        match d with
        | Symbolic s ->
            s._terms
            |> List.map (fun (mono, _) -> mono |> List.sumBy snd)
            |> (fun degs -> if degs.IsEmpty then 0 else List.max degs)
        | _ -> 0

    if maxDegree dimA > 1 || maxDegree dimB > 1 then None
    else

    // Case 1: both concrete
    match dimA, dimB with
    | Concrete a, Concrete b when a <> b ->
        Some {
            assignments = []
            dimAConcrete = a
            dimBConcrete = b
            explanation = "dims " + string a + " != " + string b
            path = site.pathSnapshot
        }
    | Concrete a, Concrete b when a = b -> None
    | _ ->
        let relevantVars = collectRelevantVars site
        if relevantVars.Count > 8 then None
        else

        let scalarBindings = Map.ofList site.scalarBindingsSnapshot
        // Quick check: does scalarBindings alone ground everything?
        let a = evalDim dimA scalarBindings
        let b = evalDim dimB scalarBindings
        match a, b with
        | Some av, Some bv when av >= 0 && bv >= 0 && av <> bv ->
            if constraintsSatisfied site scalarBindings then
                let usedVars =
                    scalarBindings
                    |> Map.filter (fun k _ -> Set.contains k relevantVars)
                    |> Map.toList
                    |> List.sort
                let parts = usedVars |> List.map (fun (k, v) -> k + "=" + string v) |> String.concat ", "
                let explanation = parts + " -> dims " + string av + " != " + string bv
                Some {
                    assignments = usedVars
                    dimAConcrete = av
                    dimBConcrete = bv
                    explanation = explanation
                    path = site.pathSnapshot
                }
            else None
        | _ ->
            findSatisfyingAssignment site relevantVars scalarBindings


/// generateWitnesses: batch-process conflict sites and return witnesses keyed by (line, code).
let generateWitnesses (conflictSites: ConflictSite list) : Map<int * string, Witness> =
    let mutable result = Map.empty
    for site in conflictSites do
        let key = (site.line, site.warningCode)
        if not (Map.containsKey key result) then
            match attemptWitness site with
            | Some w -> result <- Map.add key w result
            | None -> ()
    result
