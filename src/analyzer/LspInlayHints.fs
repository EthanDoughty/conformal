// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Inlay hint provider. Walks the IR looking for first-assignment sites
// and emits a ": matrix[3 x 4]" annotation next to the variable name,
// giving the editor a ghost-text hint that reveals the inferred shape
// without requiring the user to hover.

module LspInlayHints

open Ionide.LanguageServerProtocol.Types
open Ir
open Shapes
open Env

/// Collect inlay hints from IR statements, tracking first-seen variables.
/// rangeStart and rangeEnd are 0-based LSP line numbers.
let getInlayHints
    (irProg: Ir.Program)
    (env: Env)
    (rangeStart: int)
    (rangeEnd: int)
    : InlayHint array =

    let seen = System.Collections.Generic.HashSet<string>()
    let hints = ResizeArray<InlayHint>()

    let tryEmitHint (loc: SrcLoc) (name: string) =
        if seen.Add(name) then
            let shape = Env.get env name
            match shape with
            | Bottom | UnknownShape -> ()
            | _ ->
                let lspLine = loc.line - 1
                if lspLine >= rangeStart && lspLine <= rangeEnd then
                    let hint : InlayHint = {
                        Position    = { Line = uint32 lspLine; Character = uint32 (loc.col - 1 + name.Length) }
                        Label       = U2.C1 (": " + shapeToString shape)
                        Kind        = Some InlayHintKind.Type
                        TextEdits   = None
                        Tooltip     = None
                        PaddingLeft = Some true
                        PaddingRight = None
                        Data        = None
                    }
                    hints.Add(hint)

    let rec walkStmts (stmts: Stmt list) =
        for stmt in stmts do
            walkStmt stmt

    and walkStmt (stmt: Stmt) =
        match stmt with
        | Assign(loc, name, _) ->
            tryEmitHint loc name

        | AssignMulti(loc, targets, _) ->
            for i, name in targets |> List.indexed do
                // First target uses the AssignMulti loc; subsequent targets
                // don't have individual locs, so we skip column positioning
                // but still emit on the same line for shape visibility.
                if i = 0 then
                    tryEmitHint loc name
                else
                    // Emit with the same line but mark as seen (shape still useful)
                    tryEmitHint loc name

        | For(loc, var_, _, body) ->
            tryEmitHint loc var_
            walkStmts body

        | If(_, _, thenBody, elseBody) ->
            walkStmts thenBody
            walkStmts elseBody

        | IfChain(_, _, bodies, elseBody) ->
            for body in bodies do
                walkStmts body
            walkStmts elseBody

        | While(_, _, body) ->
            walkStmts body

        | Switch(_, _, cases, otherwise) ->
            for (_, caseBody) in cases do
                walkStmts caseBody
            walkStmts otherwise

        | Try(_, tryBody, catchBody) ->
            walkStmts tryBody
            walkStmts catchBody

        // Do NOT recurse into FunctionDef bodies (separate scope)
        | FunctionDef _ -> ()

        // All other statements: no assignments to track
        | _ -> ()

    walkStmts irProg.body
    hints.ToArray()
