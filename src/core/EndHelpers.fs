// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Resolves the end keyword inside indexing expressions. Walks BinOp
// trees to detect end references and can evaluate end-relative
// arithmetic like end-1 or end/2 when the extent is concrete.

module EndHelpers

open Ir

/// Recursively check if End appears in a BinOp tree.
let rec binopContainsEnd (expr: Expr) : bool =
    match expr with
    | End _ -> true
    | BinOp(_, _, left, right) -> binopContainsEnd left || binopContainsEnd right
    | _ -> false


/// Evaluate a BinOp tree with End resolved to endValue.
/// Returns None if can't resolve (e.g., contains variables).
let rec evalEndArithmetic (expr: Expr) (endValue: int) : int option =
    match expr with
    | End _ -> Some endValue
    | Const(_, v) -> Some (int v)
    | BinOp(_, op, left, right) ->
        match evalEndArithmetic left endValue, evalEndArithmetic right endValue with
        | Some l, Some r ->
            match op with
            | "+" -> Some (l + r)
            | "-" -> Some (l - r)
            | "*" -> Some (l * r)
            | "/" -> if r <> 0 then Some (l / r) else None
            | _   -> None
        | _ -> None
    | _ -> None   // Var or other nodes: can't resolve
