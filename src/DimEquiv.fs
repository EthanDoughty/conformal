module DimEquiv

open Shapes

// ---------------------------------------------------------------------------
// Union-find equivalence store for dimension equality classes.
// Keys are dimStr strings (matching existing constraint key format).
// Each equivalence class optionally tracks a concrete integer value.
// ---------------------------------------------------------------------------

/// Mutable union-find structure for dimension equivalence classes.
type DimEquiv = {
    parent:   System.Collections.Generic.Dictionary<string, string>
    rank:     System.Collections.Generic.Dictionary<string, int>
    concrete: System.Collections.Generic.Dictionary<string, int>
}

/// Create a fresh empty equivalence store.
let create () : DimEquiv = {
    parent   = System.Collections.Generic.Dictionary<string, string>()
    rank     = System.Collections.Generic.Dictionary<string, int>()
    concrete = System.Collections.Generic.Dictionary<string, int>()
}

/// Ensure a key exists in the parent/rank maps (lazy initialization).
let private ensureNode (eq: DimEquiv) (key: string) : unit =
    if not (eq.parent.ContainsKey(key)) then
        eq.parent.[key] <- key
        eq.rank.[key]   <- 0

/// Find canonical representative with path compression.
let rec find (eq: DimEquiv) (key: string) : string =
    ensureNode eq key
    if eq.parent.[key] = key then key
    else
        let root = find eq eq.parent.[key]
        eq.parent.[key] <- root   // path compression
        root

/// Merge equivalence classes of two dimension keys.
/// If either class has a concrete value, propagates it to the merged root.
/// Returns true if a NEW concrete propagation occurred.
let union (eq: DimEquiv) (key1: string) (key2: string) : bool =
    let r1 = find eq key1
    let r2 = find eq key2
    if r1 = r2 then false
    else
        // Try to parse keys as integers (for "5" style dim strings)
        let tryParseInt (s: string) =
            match System.Int32.TryParse(s) with
            | true, v -> Some v
            | _ -> None

        // Union by rank
        let rank1 = if eq.rank.ContainsKey(r1) then eq.rank.[r1] else 0
        let rank2 = if eq.rank.ContainsKey(r2) then eq.rank.[r2] else 0
        let newRoot, oldRoot =
            if rank1 >= rank2 then r1, r2
            else r2, r1
        eq.parent.[oldRoot] <- newRoot
        if rank1 = rank2 then
            eq.rank.[newRoot] <- (if eq.rank.ContainsKey(newRoot) then eq.rank.[newRoot] else 0) + 1

        // Propagate concrete values
        let conc1 = if eq.concrete.ContainsKey(r1) then Some eq.concrete.[r1] else tryParseInt r1
        let conc2 = if eq.concrete.ContainsKey(r2) then Some eq.concrete.[r2] else tryParseInt r2
        let merged =
            match conc1, conc2 with
            | Some v, _ -> Some v
            | _, Some v -> Some v
            | None, None -> None
        match merged with
        | Some v ->
            eq.concrete.[newRoot] <- v
            true
        | None -> false


/// Record that a dimension key has a concrete value.
/// Propagates to the class root. Returns true if this is new information.
let setConcrete (eq: DimEquiv) (key: string) (value: int) : bool =
    let root = find eq key
    if eq.concrete.ContainsKey(root) then
        false  // already known (don't overwrite; first writer wins)
    else
        eq.concrete.[root] <- value
        true

/// Query: are two dimension keys in the same equivalence class?
let equivalent (eq: DimEquiv) (key1: string) (key2: string) : bool =
    find eq key1 = find eq key2

/// Query: does the class containing this key have a known concrete value?
let getConcrete (eq: DimEquiv) (key: string) : int option =
    let root = find eq key
    // Check explicit concrete map first, then try parsing root as integer
    if eq.concrete.ContainsKey(root) then Some eq.concrete.[root]
    else
        match System.Int32.TryParse(root) with
        | true, v -> Some v
        | _ -> None

/// Snapshot current state: produce an independent copy.
/// Mutations to the original do NOT affect the snapshot, and vice versa.
let snapshot (eq: DimEquiv) : DimEquiv = {
    parent   = System.Collections.Generic.Dictionary<string, string>(eq.parent)
    rank     = System.Collections.Generic.Dictionary<string, int>(eq.rank)
    concrete = System.Collections.Generic.Dictionary<string, int>(eq.concrete)
}

/// Intersect two DimEquiv stores: keep only equivalences present in BOTH.
/// Used for branch joins. Conservative: drops any equivalence not established in all branches.
let intersect (eq1: DimEquiv) (eq2: DimEquiv) : DimEquiv =
    let result = create ()

    // Collect all unique keys across both stores
    let allKeys =
        seq {
            yield! eq1.parent.Keys
            yield! eq2.parent.Keys
        }
        |> Seq.distinct
        |> Seq.toList

    // For each pair of keys, if they are equivalent in BOTH stores, union them in result
    for i in 0 .. allKeys.Length - 1 do
        for j in i + 1 .. allKeys.Length - 1 do
            let k1 = allKeys.[i]
            let k2 = allKeys.[j]
            if equivalent eq1 k1 k2 && equivalent eq2 k1 k2 then
                union result k1 k2 |> ignore

    // Propagate concrete values that are known in BOTH stores
    for key in allKeys do
        match getConcrete eq1 key, getConcrete eq2 key with
        | Some v1, Some v2 when v1 = v2 ->
            setConcrete result key v1 |> ignore
        | _ -> ()

    result
