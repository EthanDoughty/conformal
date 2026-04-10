// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Scoped variable environment backed by immutable F# maps. Supports
// parent-pointer chains for nested function scopes: get() walks the
// parent chain on a miss, set() writes only to the local scope.

module Env

open Shapes

type Env = {
    mutable bindings:   Map<string, Shape>
    mutable dimAliases: Map<string, Dim>
    parent:             Env option
}

module Env =

    let create () : Env = {
        bindings   = Map.empty
        dimAliases = Map.empty
        parent     = None
    }

    let createWithParent (parent: Env) : Env = {
        bindings   = Map.empty
        dimAliases = Map.empty
        parent     = Some parent
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

    let hasLocal (env: Env) (name: string) : bool =
        Map.containsKey name env.bindings

    let rec contains (env: Env) (name: string) : bool =
        Map.containsKey name env.bindings ||
        (match env.parent with
         | Some p -> contains p name
         | None   -> false)

    let copy (env: Env) : Env = {
        bindings   = env.bindings
        dimAliases = env.dimAliases
        parent     = env.parent
    }

    let pushScope (env: Env) : Env = createWithParent env

    let replaceLocal (env: Env) (other: Env) : unit =
        env.bindings   <- other.bindings
        env.dimAliases <- other.dimAliases

    let localBindingsEqual (env: Env) (other: Env) : bool =
        env.bindings = other.bindings

// ---------------------------------------------------------------------------
// Module-level join/widen (mirror Python's join_env / widen_env)
// ---------------------------------------------------------------------------

let joinEnv (env1: Env) (env2: Env) : Env =
    let result = {
        bindings   = Map.empty
        dimAliases = Map.empty
        parent     = env1.parent
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
    result

let widenEnv (env1: Env) (env2: Env) : Env =
    let result = {
        bindings   = Map.empty
        dimAliases = Map.empty
        parent     = env1.parent
    }
    let allVars =
        Set.union
            (env1.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
            (env2.bindings |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
    for var in allVars |> Set.toList |> List.sort do
        let s1 = defaultArg (Map.tryFind var env1.bindings) Bottom
        let s2 = defaultArg (Map.tryFind var env2.bindings) Bottom
        Env.set result var (widenShape s1 s2)
    result
