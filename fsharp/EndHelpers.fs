module EndHelpers

open Ir

// ---------------------------------------------------------------------------
// Utilities for resolving the End keyword in indexing expressions.
// Port of analysis/end_helpers.py
// ---------------------------------------------------------------------------

/// binopContainsEnd: recursively check if End appears in a BinOp tree.
let rec binopContainsEnd (expr: Expr) : bool =
    match expr with
    | End _ -> true
    | BinOp(_, _, _, left, right) -> binopContainsEnd left || binopContainsEnd right
    | _ -> false


/// evalEndArithmetic: evaluate a BinOp tree with End resolved to endValue.
/// Returns None if can't resolve (e.g., contains variables).
let rec evalEndArithmetic (expr: Expr) (endValue: int) : int option =
    match expr with
    | End _ -> Some endValue
    | Const(_, _, v) -> Some (int v)
    | BinOp(_, _, op, left, right) ->
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
