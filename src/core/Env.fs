// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Scoped variable environment backed by immutable F# maps. Supports
// parent-pointer chains for nested function scopes: get() walks the
// parent chain on a miss, set() writes only to the local scope.

module Env

open Shapes

type Env = {
    mutable bindings:     Map<string, Shape>
    mutable dimAliases:   Map<string, Dim>
    mutable valueClasses: Map<string, VClass>
    parent:               Env option
}

module Env =

    let create () : Env = {
        bindings     = Map.empty
        dimAliases   = Map.empty
        valueClasses = Map.empty
        parent       = None
    }

    let createWithParent (parent: Env) : Env = {
        bindings     = Map.empty
        dimAliases   = Map.empty
        valueClasses = Map.empty
        parent       = Some parent
    }

    let rec get (env: Env) (name: string) : Shape =
        match Map.tryFind name env.bindings with
        | Some s -> s
        | None ->
            match env.parent with
            | Some p -> get p name
            | None   -> Bottom

    let set (env: Env) (name: string) (shape: Shape) : unit =
        env.bindings <- Map.add name shape env.bindings
        // FAILSAFE: any reassignment drops a stale value-class entry. Only the
        // dedicated class writers (setClass) opt back in.
        env.valueClasses <- Map.remove name env.valueClasses

    let hasLocal (env: Env) (name: string) : bool =
        Map.containsKey name env.bindings

    let rec contains (env: Env) (name: string) : bool =
        Map.containsKey name env.bindings ||
        (match env.parent with
         | Some p -> contains p name
         | None   -> false)

    let setClass (env: Env) (name: string) (c: VClass) : unit =
        env.valueClasses <- Map.add name c env.valueClasses

    let rec getClass (env: Env) (name: string) : VClass option =
        match Map.tryFind name env.valueClasses with
        | Some c -> Some c
        | None ->
            match env.parent with
            | Some p -> getClass p name
            | None   -> None

    let copy (env: Env) : Env = {
        bindings     = env.bindings
        dimAliases   = env.dimAliases
        valueClasses = env.valueClasses
        parent       = env.parent
    }

    let pushScope (env: Env) : Env = createWithParent env

    let replaceLocal (env: Env) (other: Env) : unit =
        env.bindings     <- other.bindings
        env.dimAliases   <- other.dimAliases
        env.valueClasses <- other.valueClasses

    let localBindingsEqual (env: Env) (other: Env) : bool =
        env.bindings = other.bindings

// --- Module-level join/widen (mirror Python's join_env / widen_env) ---

/// Intersect two value-class maps (keep only names present in both with the SAME class).
/// Same discipline as Intervals.joinUpperBounds: sound at branch merges because a variable
/// classified numeric on one path and logical on the other must not survive as either.
let private intersectClasses (a: Map<string, VClass>) (b: Map<string, VClass>) : Map<string, VClass> =
    a |> Map.filter (fun name c ->
        match Map.tryFind name b with
        | Some c2 -> c = c2
        | None -> false)

let joinEnv (env1: Env) (env2: Env) : Env =
    let result = {
        bindings     = Map.empty
        dimAliases   = Map.empty
        valueClasses = Map.empty
        parent       = env1.parent
    }
    let allVars =
        Set.union
            (env1.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
            (env2.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
    // List.sort ensures deterministic iteration order across runs.
    for var in allVars |> Set.toList |> List.sort do
        let s1 = defaultArg (Map.tryFind var env1.bindings) Bottom
        let s2 = defaultArg (Map.tryFind var env2.bindings) Bottom
        Env.set result var (joinShape s1 s2)
    // Env.set clears valueClasses as each var is written above, so the intersection
    // must be applied AFTER the loop, not before.
    result.valueClasses <- intersectClasses env1.valueClasses env2.valueClasses
    result

let widenEnv (env1: Env) (env2: Env) : Env =
    let result = {
        bindings     = Map.empty
        dimAliases   = Map.empty
        valueClasses = Map.empty
        parent       = env1.parent
    }
    let allVars =
        Set.union
            (env1.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
            (env2.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
    for var in allVars |> Set.toList |> List.sort do
        let s1 = defaultArg (Map.tryFind var env1.bindings) Bottom
        let s2 = defaultArg (Map.tryFind var env2.bindings) Bottom
        Env.set result var (widenShape s1 s2)
    result.valueClasses <- intersectClasses env1.valueClasses env2.valueClasses
    result
