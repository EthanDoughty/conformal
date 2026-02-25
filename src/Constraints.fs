module Constraints

open Ir
open Shapes
open Env
open SymDim
open Context

// ---------------------------------------------------------------------------
// Constraint tracking and path-sensitive joins for dimension equality.
// Port of analysis/constraints.py
// ---------------------------------------------------------------------------

/// tryExtractConstValue: extract integer value from a Const expression, or None.
/// Port of analysis/constraints.py try_extract_const_value().
let tryExtractConstValue (expr: Expr) : int option =
    match expr with
    | Const(_, v) when v = System.Math.Floor(v : float) -> Some (int v)
    | _ -> None


/// dimKey: structural sort key for canonicalizing constraint pairs.
let private dimKey (d: Dim) : int * string =
    match d with
    | Concrete n -> (0, string n)
    | Symbolic s -> (1, SymDim.toString s)
    | Range _    -> (3, Shapes.dimStr d)   // Range sorts after Symbolic
    | Unknown    -> (2, "")


/// isPreBoundVar: check if a Dim is a simple variable that's already bound in env.
let private isPreBoundVar (d: Dim) (env: Env) : bool =
    match d with
    | Symbolic s ->
        // Check if s is a simple variable: one term, coeff=1, single var with exp=1
        match s._terms with
        | [([varName, 1], coeff)] when coeff = Rational.One ->
            // Check if varName is bound in env (not bottom)
            not (Env.get env varName = Bottom)
        | _ -> false
    | Range _ -> false   // Range dims are not simple pre-bound variables
    | _ -> false


/// recordConstraint: record equality constraint between two dimensions.
/// Skips trivial, both-concrete, None, and pre-bound constraints.
let recordConstraint (ctx: AnalysisContext) (env: Env) (dim1: Dim) (dim2: Dim) (line: int) : unit =
    match dim1, dim2 with
    | Unknown, _ | _, Unknown -> ()
    | _ when dim1 = dim2 -> ()
    | Concrete _, Concrete _ -> ()   // already caught by dimsDefinitelyConflict
    | _ ->
        if isPreBoundVar dim1 env || isPreBoundVar dim2 env then ()
        else
            // Canonicalize: sort by dimKey to ensure (a, b) and (b, a) are same
            let k1 = dimKey dim1
            let k2 = dimKey dim2
            let canonKey1, canonKey2 =
                if k1 <= k2 then (Shapes.dimStr dim1, Shapes.dimStr dim2)
                else (Shapes.dimStr dim2, Shapes.dimStr dim1)
            let canonical = (canonKey1, canonKey2)

            ctx.cst.constraints.Add(canonical) |> ignore
            // Store provenance (first-seen: keep original line)
            if not (ctx.cst.constraintProvenance.ContainsKey(canonical)) then
                ctx.cst.constraintProvenance.[canonical] <- line


/// snapshotConstraints: return a copy of current constraint set.
let snapshotConstraints (ctx: AnalysisContext) : System.Collections.Generic.HashSet<string * string> =
    System.Collections.Generic.HashSet<string * string>(ctx.cst.constraints)


/// joinConstraints: path-sensitive join: keep baseline + constraints added in ALL branches.
let joinConstraints
    (baseline: System.Collections.Generic.HashSet<string * string>)
    (branchSets: System.Collections.Generic.HashSet<string * string> list)
    : System.Collections.Generic.HashSet<string * string> =

    // Extract new constraints per branch (branch - baseline)
    let newPerBranch =
        branchSets
        |> List.map (fun branch ->
            let newSet = System.Collections.Generic.HashSet<string * string>(branch)
            for b in baseline do newSet.Remove(b) |> ignore
            newSet)

    // Intersection of all new constraints (only if all branches added them)
    let commonNew =
        match newPerBranch with
        | [] -> System.Collections.Generic.HashSet<string * string>()
        | first :: rest ->
            let result = System.Collections.Generic.HashSet<string * string>(first)
            for s in rest do
                result.IntersectWith(s)
            result

    // Return baseline + common new
    let result = System.Collections.Generic.HashSet<string * string>(baseline)
    for c in commonNew do result.Add(c) |> ignore
    result


/// validateBinding: check if binding var_name=value conflicts with recorded constraints.
let validateBinding
    (ctx: AnalysisContext)
    (env: Env)
    (varName: string)
    (value: int)
    (warnings: ResizeArray<Diagnostics.Diagnostic>)
    (line: int)
    : unit =

    let targetDimStr = Shapes.dimStr (Symbolic (SymDim.SymDim.var varName))

    for (d1, d2) in ctx.cst.constraints do
        // Check if this constraint involves our target variable
        let otherDimStr =
            if d1 = targetDimStr then Some d2
            elif d2 = targetDimStr then Some d1
            else None

        match otherDimStr with
        | None -> ()
        | Some other ->
            // Found a constraint: check if 'other' conflicts with value
            // Try parse 'other' as integer
            match System.Int32.TryParse(other) with
            | true, otherInt ->
                if otherInt <> value then
                    let sourceLine =
                        match ctx.cst.constraintProvenance.TryGetValue((d1, d2)) with
                        | true, l -> l
                        | _ -> 0
                    warnings.Add(Diagnostics.warnConstraintConflict line varName value other sourceLine)
            | _ ->
                // Check if other dim is in scalar_bindings
                match ctx.cst.scalarBindings.TryGetValue(other) with
                | true, otherValue when otherValue <> value ->
                    let sourceLine =
                        match ctx.cst.constraintProvenance.TryGetValue((d1, d2)) with
                        | true, l -> l
                        | _ -> 0
                    warnings.Add(Diagnostics.warnConstraintConflict line varName value other sourceLine)
                | _ -> ()
