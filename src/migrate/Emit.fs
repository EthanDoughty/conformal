module Emit

open PyAst

// -------------------------------------------------------------------------
// Operator precedence (lower number = binds less tightly)
// -------------------------------------------------------------------------

let private precedence (op: string) : int =
    match op with
    | "or" -> 1
    | "and" -> 2
    | "not " -> 3
    | "==" | "!=" | "<" | ">" | "<=" | ">=" -> 4
    | "+" | "-" -> 5
    | "*" | "/" | "//" | "%" | "@" -> 6
    | "**" -> 7
    | _ -> 10  // unknown ops: high precedence (no wrapping)

let private needsParens (parentOp: string) (child: PyExpr) (isRight: bool) : bool =
    match child with
    | PyBinOp(childOp, _, _) ->
        let parentPrec = precedence parentOp
        let childPrec = precedence childOp
        if childPrec < parentPrec then true
        elif childPrec = parentPrec && isRight && parentOp <> "+" && parentOp <> "*" then true
        else false
    | _ -> false

// -------------------------------------------------------------------------
// Expression emission
// -------------------------------------------------------------------------

let rec emitExpr (expr: PyExpr) : string =
    match expr with
    | PyVar name -> name
    | PyConst v ->
        if v = System.Math.Floor(v) && not (System.Double.IsInfinity v) && not (System.Double.IsNaN v) then
            // Emit integers without decimal point
            sprintf "%g" v
        else
            sprintf "%g" v
    | PyStr s -> sprintf "'%s'" (s.Replace("\\", "\\\\").Replace("'", "\\'"))
    | PyBool true -> "True"
    | PyBool false -> "False"
    | PyNone -> "None"
    | PyBinOp("=", left, right) ->
        // Assignment expression (used for index assignment)
        sprintf "%s = %s" (emitExpr left) (emitExpr right)
    | PyBinOp(op, left, right) ->
        let l = if needsParens op left false then sprintf "(%s)" (emitExpr left) else emitExpr left
        let r = if needsParens op right true then sprintf "(%s)" (emitExpr right) else emitExpr right
        sprintf "%s %s %s" l op r
    | PyUnaryOp(op, operand) ->
        let inner = emitExpr operand
        match operand with
        | PyBinOp _ -> sprintf "%s(%s)" op inner
        | _ -> sprintf "%s%s" op inner
    | PyCall(func, args, kwargs) ->
        let sArgs = args |> List.map emitExpr
        let sKwargs = kwargs |> List.map (fun (k, v) -> sprintf "%s=%s" k (emitExpr v))
        let allArgs = sArgs @ sKwargs |> String.concat ", "
        sprintf "%s(%s)" (emitExpr func) allArgs
    | PyIndex(base_, indices) ->
        let sIndices = indices |> List.map emitIdx |> String.concat ", "
        sprintf "%s[%s]" (emitExpr base_) sIndices
    | PyAttr(base_, attr) ->
        sprintf "%s.%s" (emitExpr base_) attr
    | PyList items ->
        let sItems = items |> List.map emitExpr |> String.concat ", "
        sprintf "[%s]" sItems
    | PyTuple items ->
        match items with
        | [single] -> sprintf "(%s,)" (emitExpr single)
        | _ ->
            let sItems = items |> List.map emitExpr |> String.concat ", "
            sprintf "(%s)" sItems
    | PyArray rows ->
        let sRows = rows |> List.map (fun row ->
            let sElems = row |> List.map emitExpr |> String.concat ", "
            sprintf "[%s]" sElems)
        sprintf "np.array([%s])" (sRows |> String.concat ", ")
    | PyLambda(parms, body) ->
        sprintf "lambda %s: %s" (parms |> String.concat ", ") (emitExpr body)
    | PyComment text ->
        // Inline comment attached to an expression: emit as-is
        // (the surrounding expression handles placement)
        sprintf "# %s" text

and private emitIdx (idx: PyIdx) : string =
    match idx with
    | PyScalarIdx expr -> emitExpr expr
    | PySlice(lo, hi, step) ->
        let sLo = lo |> Option.map emitExpr |> Option.defaultValue ""
        let sHi = hi |> Option.map emitExpr |> Option.defaultValue ""
        match step with
        | Some s -> sprintf "%s:%s:%s" sLo sHi (emitExpr s)
        | None -> sprintf "%s:%s" sLo sHi

// -------------------------------------------------------------------------
// Statement emission
// -------------------------------------------------------------------------

let rec emitStmt (indent: int) (stmt: PyStmt) : string list =
    let pad = System.String(' ', indent)
    match stmt with
    | PyAssign(target, PyComment comment) ->
        // Expression with trailing comment — emit as assignment + comment
        [sprintf "%s%s = None  # %s" pad target comment]
    | PyAssign(target, expr) ->
        [sprintf "%s%s = %s" pad target (emitExpr expr)]
    | PyMultiAssign(targets, expr) ->
        [sprintf "%s%s = %s" pad (targets |> String.concat ", ") (emitExpr expr)]
    | PyExprStmt(PyBinOp("=", left, right)) ->
        // Index assignment: A[i] = expr
        [sprintf "%s%s = %s" pad (emitExpr left) (emitExpr right)]
    | PyExprStmt expr ->
        // Check for inline comment on the expression
        match expr with
        | PyBinOp(op, left, PyComment comment) ->
            [sprintf "%s%s %s  # %s" pad (emitExpr left) op comment]
        | _ ->
            [sprintf "%s%s" pad (emitExpr expr)]
    | PyIf(cond, thenBody, elifs, elseBody) ->
        let ifLine = [sprintf "%sif %s:" pad (emitExpr cond)]
        let thenLines = thenBody |> List.collect (emitStmt (indent + 4))
        let thenLines = if thenLines.IsEmpty then [sprintf "%s    pass" pad] else thenLines
        let elifLines =
            elifs |> List.collect (fun (c, b) ->
                let eLine = [sprintf "%selif %s:" pad (emitExpr c)]
                let bLines = b |> List.collect (emitStmt (indent + 4))
                let bLines = if bLines.IsEmpty then [sprintf "%s    pass" pad] else bLines
                eLine @ bLines)
        let elseLines =
            if elseBody.IsEmpty then []
            else
                [sprintf "%selse:" pad]
                @ (elseBody |> List.collect (emitStmt (indent + 4)))
        ifLine @ thenLines @ elifLines @ elseLines
    | PyFor(var_, iterable, body) ->
        let forLine = [sprintf "%sfor %s in %s:" pad var_ (emitExpr iterable)]
        let bodyLines = body |> List.collect (emitStmt (indent + 4))
        let bodyLines = if bodyLines.IsEmpty then [sprintf "%s    pass" pad] else bodyLines
        forLine @ bodyLines
    | PyWhile(cond, body) ->
        let whileLine = [sprintf "%swhile %s:" pad (emitExpr cond)]
        let bodyLines = body |> List.collect (emitStmt (indent + 4))
        let bodyLines = if bodyLines.IsEmpty then [sprintf "%s    pass" pad] else bodyLines
        whileLine @ bodyLines
    | PyFuncDef(name, parms, body, _returnVars) ->
        let defLine = [sprintf "%sdef %s(%s):" pad name (parms |> String.concat ", ")]
        let bodyLines = body |> List.collect (emitStmt (indent + 4))
        let bodyLines = if bodyLines.IsEmpty then [sprintf "%s    pass" pad] else bodyLines
        defLine @ bodyLines
    | PyReturn exprs ->
        match exprs with
        | [] -> [sprintf "%sreturn" pad]
        | _ -> [sprintf "%sreturn %s" pad (exprs |> List.map emitExpr |> String.concat ", ")]
    | PyCommentStmt text ->
        [sprintf "%s# %s" pad text]
    | PyImport(module_, Some alias) ->
        [sprintf "%simport %s as %s" pad module_ alias]
    | PyImport(module_, None) ->
        [sprintf "%simport %s" pad module_]
    | PyFromImport(module_, names) ->
        [sprintf "%sfrom %s import %s" pad module_ (names |> String.concat ", ")]
    | PyPass -> [sprintf "%spass" pad]
    | PyBreak -> [sprintf "%sbreak" pad]
    | PyContinue -> [sprintf "%scontinue" pad]
    | PyTry(tryBody, exceptBody) ->
        let tryLine = [sprintf "%stry:" pad]
        let tryLines = tryBody |> List.collect (emitStmt (indent + 4))
        let tryLines = if tryLines.IsEmpty then [sprintf "%s    pass" pad] else tryLines
        let exceptLine = [sprintf "%sexcept Exception:" pad]
        let exceptLines = exceptBody |> List.collect (emitStmt (indent + 4))
        let exceptLines = if exceptLines.IsEmpty then [sprintf "%s    pass" pad] else exceptLines
        tryLine @ tryLines @ exceptLine @ exceptLines

// -------------------------------------------------------------------------
// Program emission
// -------------------------------------------------------------------------

let emitProgram (program: PyProgram) : string list =
    let importLines = program.imports |> List.collect (emitStmt 0)
    let bodyLines = program.body |> List.collect (emitStmt 0)
    let separator = if bodyLines.IsEmpty then [] else [""]
    importLines @ separator @ bodyLines
