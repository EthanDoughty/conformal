// Conformal Migrate: MATLAB-to-Python Transpiler
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Copy semantics analysis. MATLAB has pass-by-value semantics while
// Python uses reference semantics, so whenever a MATLAB assignment
// would have forked state (y = x followed by in-place mutation of y),
// the transpiler must insert an explicit .copy() to preserve meaning.
// This module finds all the source locations where that is needed.

module CopySemantics

open Ir
open Shapes

/// Returns set of SrcLocs where .copy() should be inserted on the RHS.
/// Conservative: may insert unnecessary copies, but never misses a needed one.
///
/// Algorithm:
/// 1. Collect all variable names that are targets of IndexAssign, CellAssign,
///    StructAssign, IndexStructAssign, or FieldIndexAssign (the "mutated set").
/// 2. For each Assign(loc, name, Var(srcLoc, srcName)) where srcName is in
///    the mutated set OR name is in the mutated set, and the shape at srcLoc
///    is Matrix or Cell (not Scalar), mark srcLoc as needing .copy().
let findCopySites
    (stmts: Stmt list)
    (shapeAnnotations: System.Collections.Generic.Dictionary<SrcLoc, Shape>)
    : Set<SrcLoc> =

    // Pass 1: collect names that get element-mutated anywhere in the program
    let mutatedNames = System.Collections.Generic.HashSet<string>()

    let rec scanForMutations (ss: Stmt list) =
        for s in ss do
            match s with
            | IndexAssign(_, baseName, _, _) -> mutatedNames.Add(baseName) |> ignore
            | CellAssign(_, baseName, _, _)  -> mutatedNames.Add(baseName) |> ignore
            | StructAssign(_, baseName, _, _) -> mutatedNames.Add(baseName) |> ignore
            | IndexStructAssign(_, baseName, _, _, _, _) -> mutatedNames.Add(baseName) |> ignore
            | FieldIndexAssign(_, baseName, _, _, _, _, _) -> mutatedNames.Add(baseName) |> ignore
            | LhsAssign(_, baseName, _, _) -> mutatedNames.Add(baseName) |> ignore
            | If(_, _, thenBody, elseBody) ->
                scanForMutations thenBody
                scanForMutations elseBody
            | IfChain(_, _, bodies, elseBody) ->
                for b in bodies do scanForMutations b
                scanForMutations elseBody
            | While(_, _, body) -> scanForMutations body
            | For(_, _, _, body) -> scanForMutations body
            | Switch(_, _, cases, otherwise) ->
                for (_, body) in cases do scanForMutations body
                scanForMutations otherwise
            | Try(_, tryBody, catchBody) ->
                scanForMutations tryBody
                scanForMutations catchBody
            | FunctionDef(_, _, _, _, body, _) -> scanForMutations body
            | _ -> ()

    scanForMutations stmts

    // Pass 2: find Assign(_, name, Var(srcLoc, _)) where name will be mutated
    // and the source shape is non-scalar
    let copySites = System.Collections.Generic.HashSet<SrcLoc>()

    let rec scanForCopies (ss: Stmt list) =
        for s in ss do
            match s with
            | Assign(_, name, Var(srcLoc, _srcName)) when mutatedNames.Contains(name) ->
                match shapeAnnotations.TryGetValue(srcLoc) with
                | true, shape when isMatrix shape || isCell shape -> copySites.Add(srcLoc) |> ignore
                | _ -> ()
            | If(_, _, thenBody, elseBody) ->
                scanForCopies thenBody
                scanForCopies elseBody
            | IfChain(_, _, bodies, elseBody) ->
                for b in bodies do scanForCopies b
                scanForCopies elseBody
            | While(_, _, body) -> scanForCopies body
            | For(_, _, _, body) -> scanForCopies body
            | Switch(_, _, cases, otherwise) ->
                for (_, body) in cases do scanForCopies body
                scanForCopies otherwise
            | Try(_, tryBody, catchBody) ->
                scanForCopies tryBody
                scanForCopies catchBody
            | FunctionDef(_, _, _, _, body, _) -> scanForCopies body
            | _ -> ()

    scanForCopies stmts
    Set.ofSeq copySites
