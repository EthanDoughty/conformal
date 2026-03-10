module PathConstraints

open Ir

// ---------------------------------------------------------------------------
// Condition formatting helpers (mirrors path_constraints.py)
// ---------------------------------------------------------------------------

let private formatSimple (expr: Expr) : string option =
    match expr with
    | Var(_, name)  -> Some name
    | Const(_, v)   ->
        if v = System.Math.Floor(v : float) then Some (string (int64 v))
        else Some (string v)
    | Neg(_, Const(_, v)) ->
        if v = System.Math.Floor(v : float) then Some (string -(int64 v))
        else Some (string -v)
    | _ -> None

let private formatConditionExpr (expr: Expr) : string =
    match expr with
    | BinOp(_, op, left, right)
        when Set.contains op (Set.ofList [">"; ">="; "<"; "<="; "=="; "~="]) ->
        match formatSimple left, formatSimple right with
        | Some l, Some r -> l + " " + op + " " + r
        | _              -> "condition at line " + string expr.Line
    | _ -> "condition at line " + string expr.Line

// ---------------------------------------------------------------------------
// PathConstraintStack: branch condition tracking
// Each entry: (condition_description, branch_taken, line)
// ---------------------------------------------------------------------------

type PathConstraintEntry = {
    description: string
    branchTaken: bool
    line:        int
}

type PathConstraintStack() =
    let mutable _stack : PathConstraintEntry list = []

    member _.Push(conditionExpr: Expr, branchTaken: bool, line: int) =
        let desc = formatConditionExpr conditionExpr
        _stack <- { description = desc; branchTaken = branchTaken; line = line } :: _stack

    member _.Pop() =
        match _stack with
        | _ :: rest -> _stack <- rest
        | []        -> ()  // defensive no-op

    member _.Snapshot() : PathConstraintEntry list =
        List.rev _stack   // oldest first, matching Python tuple order

    member _.Stack = _stack
