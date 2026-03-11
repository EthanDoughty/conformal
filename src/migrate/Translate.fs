module Translate

open Ir
open Shapes
open PyAst
open BuiltinMap

type TranslateContext = {
    shapeAnnotations: System.Collections.Generic.Dictionary<SrcLoc, Shape>
    copySites: Set<SrcLoc>
    env: Env.Env
    mutable usedImports: Set<string>
    /// Current function's output variable names (for return translation)
    mutable currentReturnVars: string list
}

let lookupShape (tctx: TranslateContext) (loc: SrcLoc) : Shape option =
    match tctx.shapeAnnotations.TryGetValue(loc) with
    | true, shape -> Some shape
    | false, _ -> None

/// Infer the shape of an expression using annotations, env, and simple rules.
let inferExprShape (tctx: TranslateContext) (expr: Expr) : Shape option =
    // Try annotation map first
    match lookupShape tctx expr.Loc with
    | Some s -> Some s
    | None ->
        // Fall back to type inference from expression structure
        match expr with
        | Const _ -> Some Scalar
        | StringLit _ -> Some StringShape
        | Var(_, name) ->
            let s = Env.Env.get tctx.env name
            if s = Bottom then None else Some s
        | Neg(_, _) -> Some Scalar  // conservative
        | _ -> None

/// Check if shape is definitely a matrix (2D non-scalar)
let private isMatrixShape (s: Shape) =
    match s with Matrix _ -> true | _ -> false

/// Filter out synthetic parser sentinels (ExprStmt at line 0 with Const 0)
let private isSyntheticSentinel (stmt: Stmt) =
    match stmt with
    | ExprStmt({ line = 0; col = 0 }, Const({ line = 0; col = 0 }, 0.0)) -> true
    | _ -> false

// -------------------------------------------------------------------------
// Expression translation
// -------------------------------------------------------------------------

let rec translateExpr (expr: Expr) (tctx: TranslateContext) : PyExpr =
    tctx.usedImports <- Set.add "numpy" tctx.usedImports
    match expr with
    | Var(_, name) ->
        // MATLAB true/false -> Python True/False
        match name with
        | "true" -> PyBool true
        | "false" -> PyBool false
        | "pi" -> PyAttr(PyVar "np", "pi")
        | "inf" | "Inf" -> PyAttr(PyVar "np", "inf")
        | "nan" | "NaN" -> PyAttr(PyVar "np", "nan")
        | "eps" -> PyCall(PyAttr(PyVar "np", "finfo"), [PyAttr(PyVar "np", "float64")], [])
        | _ -> PyVar name
    | Const(_, v) -> PyConst v
    | StringLit(_, s) -> PyStr s
    | Neg(_, operand) -> PyUnaryOp("-", translateExpr operand tctx)
    | Not(_, operand) ->
        let pyOp = translateExpr operand tctx
        let shape = inferExprShape tctx operand
        match shape with
        | Some s when isMatrixShape s -> PyCall(PyVar "np.logical_not", [pyOp], [])
        | _ -> PyUnaryOp("not ", pyOp)
    | BinOp(_, op, left, right) -> translateBinOp op left right tctx
    | Transpose(_, operand) ->
        PyAttr(translateExpr operand tctx, "T")
    | FieldAccess(_, base_, field) ->
        PyAttr(translateExpr base_ tctx, field)
    | Lambda(_, parms, body) ->
        PyLambda(parms, translateExpr body tctx)
    | FuncHandle(_, name) ->
        // @func -> func (Python functions are first-class)
        PyVar name
    | End(_) ->
        // 'end' in MATLAB indexing — context-dependent, emit as -1 for Python
        PyConst -1.0
    | Apply(_, base_, args) ->
        translateApply expr base_ args tctx
    | CurlyApply(_, base_, args) ->
        // Cell indexing: A{i} -> A[i-1] (same as regular indexing for migration)
        let pyBase = translateExpr base_ tctx
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        PyIndex(pyBase, pyIndices)
    | MatrixLit(_, rows) ->
        translateMatrixLit rows tctx
    | CellLit(_, rows) ->
        // Cell array -> Python list of lists
        let pyRows = rows |> List.map (fun row -> row |> List.map (fun e -> translateExpr e tctx))
        match pyRows with
        | [single] -> PyList single
        | _ -> PyList (pyRows |> List.map PyList)
    | MetaClass(_, name) ->
        PyStr name  // ?ClassName -> 'ClassName' string

and private translateBinOp (op: string) (left: Expr) (right: Expr) (tctx: TranslateContext) : PyExpr =
    let pyL () = translateExpr left tctx
    let pyR () = translateExpr right tctx
    match op with
    | "*" ->
        let lShape = inferExprShape tctx left
        let rShape = inferExprShape tctx right
        match lShape, rShape with
        | Some Scalar, Some Scalar -> PyBinOp("*", pyL(), pyR())
        | Some s1, Some s2 when isMatrixShape s1 || isMatrixShape s2 ->
            PyCall(PyVar "np.dot", [pyL(); pyR()], [])
        | Some Scalar, _ | _, Some Scalar ->
            PyBinOp("*", pyL(), pyR())
        | _ ->
            // Shape unknown: emit element-wise (conservative default)
            PyBinOp("*", pyL(), pyR())
    | ".*" -> PyBinOp("*", pyL(), pyR())
    | "./" -> PyBinOp("/", pyL(), pyR())
    | ".\\" -> PyBinOp("/", pyR(), pyL())  // a.\b = b./a
    | ".^" -> PyBinOp("**", pyL(), pyR())
    | "/" ->
        let lShape = inferExprShape tctx left
        let rShape = inferExprShape tctx right
        match lShape, rShape with
        | Some s1, Some s2 when isMatrixShape s1 && isMatrixShape s2 ->
            PyCall(PyVar "np.dot", [pyL(); PyCall(PyVar "np.linalg.inv", [pyR()], [])], [])
        | _ -> PyBinOp("/", pyL(), pyR())
    | "\\" ->
        let lShape = inferExprShape tctx left
        let rShape = inferExprShape tctx right
        match lShape, rShape with
        | Some s1, Some s2 when isMatrixShape s1 && isMatrixShape s2 ->
            PyCall(PyAttr(PyVar "np.linalg", "solve"), [pyL(); pyR()], [])
        | _ -> PyBinOp("/", pyR(), pyL())
    | "^" ->
        let lShape = inferExprShape tctx left
        match lShape with
        | Some s when isMatrixShape s ->
            PyCall(PyAttr(PyVar "np.linalg", "matrix_power"), [pyL(); pyR()], [])
        | _ -> PyBinOp("**", pyL(), pyR())
    | "~=" -> PyBinOp("!=", pyL(), pyR())
    | "&&" -> PyBinOp("and", pyL(), pyR())
    | "||" -> PyBinOp("or", pyL(), pyR())
    | "&" -> PyCall(PyVar "np.logical_and", [pyL(); pyR()], [])
    | "|" -> PyCall(PyVar "np.logical_or", [pyL(); pyR()], [])
    | _ -> PyBinOp(op, pyL(), pyR())

and private translateApply (expr: Expr) (base_: Expr) (args: IndexArg list) (tctx: TranslateContext) : PyExpr =
    match base_ with
    | Var(_, fname) ->
        // Check if this is a builtin function call
        match tryMapBuiltin fname with
        | Some mapping -> translateBuiltinCall mapping fname args tctx
        | None ->
            // Disambiguate function call vs indexing using env shape
            let varShape = Env.Env.get tctx.env fname
            if isMatrix varShape || isCell varShape then
                // Array/cell indexing
                let pyBase = PyVar fname
                let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
                PyIndex(pyBase, pyIndices)
            else
                // Treat as function call
                let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
                PyCall(PyVar fname, pyArgs, [])
    | _ ->
        // Complex base expression: treat as indexing
        let pyBase = translateExpr base_ tctx
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        PyIndex(pyBase, pyIndices)

and private translateBuiltinCall (mapping: BuiltinMapping) (fname: string) (args: IndexArg list) (tctx: TranslateContext) : PyExpr =
    let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
    let kwargs = if mapping.needsOrderF then [("order", PyStr "F")] else []

    match mapping.argTransform with
    | Direct ->
        PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | TupleFirstN n ->
        if fname = "reshape" && pyArgs.Length >= 2 then
            // reshape(A, m, n) -> np.reshape(A, (m, n), order='F')
            let arr = pyArgs.[0]
            let dims = pyArgs.[1..] |> PyTuple
            PyCall(PyVar mapping.pythonFunc, [arr; dims], kwargs)
        elif pyArgs.Length >= n then
            let tupled = pyArgs.[0..n-1] |> PyTuple
            let rest = pyArgs.[n..]
            PyCall(PyVar mapping.pythonFunc, tupled :: rest, kwargs)
        else
            PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | AttrStyle attr ->
        match pyArgs with
        | [obj] -> PyAttr(obj, attr)
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | AttrStyleDim ->
        // size(A, d) -> A.shape[d-1]
        match pyArgs with
        | [obj] -> PyAttr(obj, "shape")
        | [obj; dim] -> PyIndex(PyAttr(obj, "shape"), [PyScalarIdx(PyBinOp("-", dim, PyConst 1.0))])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | MaxShape ->
        // length(A) -> max(A.shape)
        match pyArgs with
        | [obj] -> PyCall(PyVar "max", [PyAttr(obj, "shape")], [])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | SizeAttr ->
        // numel(A) -> A.size
        match pyArgs with
        | [obj] -> PyAttr(obj, "size")
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)

and private translateCallArg (arg: IndexArg) (tctx: TranslateContext) : PyExpr =
    match arg with
    | IndexExpr(_, expr) -> translateExpr expr tctx
    | Colon _ -> PyStr ":"  // rare in function calls
    | Ir.Range(_, start, end_) ->
        PyCall(PyVar "range", [translateExpr start tctx; PyBinOp("+", translateExpr end_ tctx, PyConst 1.0)], [])
    | Ir.SteppedRange(_, start, step, end_) ->
        PyCall(PyVar "range", [translateExpr start tctx; PyBinOp("+", translateExpr end_ tctx, PyConst 1.0); translateExpr step tctx], [])

and private translateIndexArg (arg: IndexArg) (tctx: TranslateContext) : PyIdx =
    match arg with
    | Colon _ -> PySlice(None, None, None)  // : -> :
    | IndexExpr(_, expr) ->
        // A(i) -> A[i-1]
        PyScalarIdx(PyBinOp("-", translateExpr expr tctx, PyConst 1.0))
    | Ir.Range(_, start, end_) ->
        // A(a:b) -> A[a-1:b] (0-based start, exclusive end cancels)
        let pyStart = PyBinOp("-", translateExpr start tctx, PyConst 1.0)
        let pyEnd = translateExpr end_ tctx
        PySlice(Some pyStart, Some pyEnd, None)
    | Ir.SteppedRange(_, start, step, end_) ->
        let pyStart = PyBinOp("-", translateExpr start tctx, PyConst 1.0)
        let pyEnd = translateExpr end_ tctx
        let pyStep = translateExpr step tctx
        PySlice(Some pyStart, Some pyEnd, Some pyStep)

and private translateMatrixLit (rows: Expr list list) (tctx: TranslateContext) : PyExpr =
    let pyRows = rows |> List.map (fun row -> row |> List.map (fun e -> translateExpr e tctx))
    match pyRows with
    | [[single]] -> single  // [x] -> x (scalar)
    | [row] -> PyCall(PyVar "np.array", [PyList row], [])  // [a, b, c] -> np.array([a, b, c])
    | [] -> PyCall(PyVar "np.array", [PyList []], [])  // [] -> np.array([])
    | _ -> PyCall(PyVar "np.array", [PyList (pyRows |> List.map PyList)], [])

// -------------------------------------------------------------------------
// Statement translation
// -------------------------------------------------------------------------

let rec translateStmt (stmt: Stmt) (tctx: TranslateContext) : PyStmt list =
    // Filter out synthetic parser sentinels
    if isSyntheticSentinel stmt then [] else
    match stmt with
    | Assign(_, name, Var(srcLoc, srcName)) when Set.contains srcLoc tctx.copySites ->
        // Copy semantics: B = A -> B = A.copy()
        [PyAssign(name, PyCall(PyAttr(PyVar srcName, "copy"), [], []))]
    | Assign(_, name, expr) ->
        [PyAssign(name, translateExpr expr tctx)]
    | AssignMulti(_, targets, expr) ->
        [PyMultiAssign(targets, translateExpr expr tctx)]
    | ExprStmt(_, expr) ->
        [PyExprStmt(translateExpr expr tctx)]

    | If(_, cond, thenBody, elseBody) ->
        let pyCond = translateExpr cond tctx
        let pyThen = thenBody |> List.collect (fun s -> translateStmt s tctx)
        let pyElse = elseBody |> List.collect (fun s -> translateStmt s tctx)
        [PyIf(pyCond, pyThen, [], pyElse)]

    | IfChain(_, conditions, bodies, elseBody) ->
        match conditions, bodies with
        | firstCond :: restConds, firstBody :: restBodies ->
            let pyFirstCond = translateExpr firstCond tctx
            let pyFirstBody = firstBody |> List.collect (fun s -> translateStmt s tctx)
            let elifs =
                List.zip restConds restBodies
                |> List.map (fun (c, b) ->
                    (translateExpr c tctx, b |> List.collect (fun s -> translateStmt s tctx)))
            let pyElse = elseBody |> List.collect (fun s -> translateStmt s tctx)
            [PyIf(pyFirstCond, pyFirstBody, elifs, pyElse)]
        | _ -> []

    | While(_, cond, body) ->
        let pyCond = translateExpr cond tctx
        let pyBody = body |> List.collect (fun s -> translateStmt s tctx)
        [PyWhile(pyCond, pyBody)]

    | For(_, var_, it, body) ->
        let pyBody = body |> List.collect (fun s -> translateStmt s tctx)
        match it with
        | BinOp(_, ":", start, end_) ->
            // for i = a:b -> for i in range(a, b + 1):
            let pyStart = translateExpr start tctx
            let pyEnd = PyBinOp("+", translateExpr end_ tctx, PyConst 1.0)
            [PyFor(var_, PyCall(PyVar "range", [pyStart; pyEnd], []), pyBody)]
        | Apply(_, Var(_, "colon"), _) ->
            // Fallback for colon() calls
            [PyCommentStmt (sprintf "CONFORMAL: complex for-loop iterator for '%s'" var_)]
             @ [PyFor(var_, translateExpr it tctx, pyBody)]
        | _ ->
            // General iterator expression (could be Range IR node)
            let pyIt = translateForIterator it tctx
            [PyFor(var_, pyIt, pyBody)]

    | Switch(_, expr, cases, otherwise) ->
        let pyExpr = translateExpr expr tctx
        match cases with
        | [] -> []
        | (firstVal, firstBody) :: restCases ->
            let pyFirstCond = PyBinOp("==", pyExpr, translateExpr firstVal tctx)
            let pyFirstBody = firstBody |> List.collect (fun s -> translateStmt s tctx)
            let elifs =
                restCases |> List.map (fun (v, b) ->
                    (PyBinOp("==", pyExpr, translateExpr v tctx),
                     b |> List.collect (fun s -> translateStmt s tctx)))
            let pyElse = otherwise |> List.collect (fun s -> translateStmt s tctx)
            [PyIf(pyFirstCond, pyFirstBody, elifs, pyElse)]

    | Try(_, tryBody, catchBody) ->
        let pyTry = tryBody |> List.collect (fun s -> translateStmt s tctx)
        let pyCatch = catchBody |> List.collect (fun s -> translateStmt s tctx)
        [PyTry(pyTry, if pyCatch.IsEmpty then [PyPass] else pyCatch)]

    | Break _ -> [PyBreak]
    | Continue _ -> [PyContinue]
    | Return _ ->
        match tctx.currentReturnVars with
        | [] -> [PyReturn []]
        | vars -> [PyReturn (vars |> List.map PyVar)]

    | IndexAssign(_, baseName, args, expr) ->
        // A(i,j) = expr -> A[i-1, j-1] = expr
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        let pyExpr = translateExpr expr tctx
        [PyExprStmt(PyBinOp("=", PyIndex(PyVar baseName, pyIndices), pyExpr))]

    | StructAssign(_, baseName, fields, expr) ->
        let pyExpr = translateExpr expr tctx
        let target = fields |> List.fold (fun acc f -> sprintf "%s.%s" acc f) baseName
        [PyAssign(target, pyExpr)]

    | CellAssign(_, baseName, args, expr) ->
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        let pyExpr = translateExpr expr tctx
        [PyExprStmt(PyBinOp("=", PyIndex(PyVar baseName, pyIndices), pyExpr))]

    | IndexStructAssign(_, baseName, indexArgs, _, fields, expr) ->
        let pyExpr = translateExpr expr tctx
        [PyCommentStmt (sprintf "CONFORMAL: complex index-struct assignment to %s" baseName)]
         @ [PyAssign(baseName, pyExpr)]

    | FieldIndexAssign(_, baseName, prefixFields, indexArgs, _, suffixFields, expr) ->
        let pyExpr = translateExpr expr tctx
        [PyCommentStmt (sprintf "CONFORMAL: complex field-index assignment to %s" baseName)]
         @ [PyAssign(baseName, pyExpr)]

    | OpaqueStmt(_, targets, raw) ->
        let comment = [PyCommentStmt (sprintf "MATLAB: %s" (raw.Trim()))]
        match targets with
        | [] -> comment
        | _ ->
            (targets |> List.map (fun t -> PyAssign(t, PyNone)))
            @ comment

    | FunctionDef(_, name, parms, outputVars, body, _) ->
        let savedReturnVars = tctx.currentReturnVars
        tctx.currentReturnVars <- outputVars
        let pyBody = body |> List.collect (fun s -> translateStmt s tctx)
        // Add implicit return if the body doesn't end with an explicit return
        let needsReturn =
            not outputVars.IsEmpty &&
            (pyBody.IsEmpty || (match List.last pyBody with PyReturn _ -> false | _ -> true))
        let fullBody =
            if needsReturn then
                pyBody @ [PyReturn (outputVars |> List.map PyVar)]
            else
                pyBody
        tctx.currentReturnVars <- savedReturnVars
        [PyFuncDef(name, parms, (if fullBody.IsEmpty then [PyPass] else fullBody), outputVars)]

and private translateForIterator (it: Expr) (tctx: TranslateContext) : PyExpr =
    match it with
    | Apply(_, Var(_, "colon"), args) ->
        // colon(a, b) or colon(a, step, b)
        let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
        match pyArgs with
        | [a; b] -> PyCall(PyVar "range", [a; PyBinOp("+", b, PyConst 1.0)], [])
        | [a; s; b] -> PyCall(PyVar "range", [a; PyBinOp("+", b, PyConst 1.0); s], [])
        | _ -> translateExpr it tctx
    | _ ->
        // General expression as iterator
        translateExpr it tctx

// -------------------------------------------------------------------------
// Program translation
// -------------------------------------------------------------------------

let translateProgram (program: Ir.Program) (tctx: TranslateContext) (sourceFile: string) : PyProgram =
    let body = program.body |> List.collect (fun s -> translateStmt s tctx)

    let imports = [
        PyCommentStmt (sprintf "Generated by conformal-migrate 3.4.0")
        PyCommentStmt (sprintf "Source: %s" sourceFile)
        PyImport("numpy", Some "np")
    ]

    { imports = imports; body = body }
