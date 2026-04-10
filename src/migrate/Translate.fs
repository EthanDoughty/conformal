// Conformal Migrate: MATLAB-to-Python Transpiler
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Core translation pass from Conformal's MATLAB IR to the Python AST.
// Uses shape information from the analyzer to pick the right numpy
// operator (np.dot for matrix multiply, * for element-wise), resolve
// 1-to-0 index offsets, rename Python-reserved identifiers, and turn
// varargin/varargout into *args/return tuples.

module Translate

open Ir
open Shapes
open PyAst
open BuiltinMap

/// Python 3 reserved keywords that cannot be used as identifiers.
let private pythonKeywords = set [
    "False"; "None"; "True"; "and"; "as"; "assert"; "async"; "await"
    "break"; "class"; "continue"; "def"; "del"; "elif"; "else"; "except"
    "finally"; "for"; "from"; "global"; "if"; "import"; "in"; "is"
    "lambda"; "nonlocal"; "not"; "or"; "pass"; "raise"; "return"
    "try"; "while"; "with"; "yield"
]

/// Rename a MATLAB identifier if it collides with a Python keyword.
let private safeName (name: string) : string =
    if Set.contains name pythonKeywords then name + "_"
    else name

/// Constant-folding binary op constructor: folds arithmetic on two constants at build time.
let private mkBinOp (op: string) (left: PyExpr) (right: PyExpr) : PyExpr =
    match op, left, right with
    | "+", PyConst a, PyConst b -> PyConst(a + b)
    | "-", PyConst a, PyConst b -> PyConst(a - b)
    | "*", PyConst a, PyConst b -> PyConst(a * b)
    | "/", PyConst a, PyConst b when b <> 0.0 -> PyConst(a / b)
    | _ -> PyBinOp(op, left, right)

type TranslateContext = {
    shapeAnnotations: System.Collections.Generic.Dictionary<SrcLoc, Shape>
    copySites: Set<SrcLoc>
    env: Env.Env
    mutable usedImports: Set<string>
    /// Current function's output variable names (for return translation)
    mutable currentReturnVars: string list
    /// Function nesting depth (0 = top level, 1 = inside function, 2+ = nested function)
    mutable functionDepth: int
    /// Directories discovered via addpath() calls (for future cross-file import resolution)
    mutable addpathDirs: string list
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
        | "eps" -> PyAttr(PyCall(PyAttr(PyVar "np", "finfo"), [PyAttr(PyVar "np", "float64")], []), "eps")
        | "varargin" -> PyVar "args"  // bare varargin reference -> args
        | _ -> PyVar (safeName name)
    | Const(_, v) -> PyConst v
    | StringLit(_, s) -> PyStr s
    | Neg(_, operand) -> PyUnaryOp("-", translateExpr operand tctx)
    | Not(_, operand) ->
        PyCall(PyVar "np.logical_not", [translateExpr operand tctx], [])
    | BinOp(_, op, left, right) -> translateBinOp op left right tctx
    | Transpose(_, operand) ->
        PyAttr(translateExpr operand tctx, "T")
    | FieldAccess(_, base_, field) ->
        if field = "<dynamic>" then
            // Dynamic field access s.(expr) — expression lost in IR, emit getattr placeholder
            PyCall(PyVar "getattr", [translateExpr base_ tctx; PyStr "<dynamic>"], [])
        else
            PyAttr(translateExpr base_ tctx, safeName field)
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
    | CurlyApply(_, Var(_, "varargin"), args) ->
        // varargin{i} -> args[i-1]
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        PyIndex(PyVar "args", pyIndices)
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
        | Some s1, Some s2 when isMatrixShape s1 && isMatrixShape s2 ->
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
    | ":" ->
        // Colon expression outside for-loop/indexing: a:b -> np.arange(a, b + 1)
        // Stepped: (a:step):b (left-assoc) -> np.arange(a, b + 1, step)
        match left with
        | BinOp(_, ":", start, step) ->
            let pyStart = translateExpr start tctx
            let pyStep = translateExpr step tctx
            let pyEnd = mkBinOp "+" (pyR()) (PyConst 1.0)
            PyCall(PyVar "np.arange", [pyStart; pyEnd; pyStep], [])
        | _ ->
            let pyEnd = mkBinOp "+" (pyR()) (PyConst 1.0)
            PyCall(PyVar "np.arange", [pyL(); pyEnd], [])
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
            // Structural hint: if any arg is Colon/Range/SteppedRange, it's definitely indexing
            let hasIndexingArg = args |> List.exists (fun a -> match a with Colon _ | Ir.Range _ | Ir.SteppedRange _ -> true | _ -> false)
            if isMatrix varShape || isCell varShape || hasIndexingArg then
                // Array/cell indexing
                let pyBase = PyVar fname
                match args with
                | [Colon _] ->
                    // A(:) -> A.ravel()  (flatten to 1D)
                    PyCall(PyAttr(pyBase, "ravel"), [], [])
                | _ ->
                    let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
                    PyIndex(pyBase, pyIndices)
            else
                // Treat as function call
                let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
                PyCall(PyVar fname, pyArgs, [])
    | _ ->
        // Complex base expression (e.g. obj.method(args))
        let pyBase = translateExpr base_ tctx
        match args with
        | [] -> PyCall(pyBase, [], [])  // Zero-arg method call: obj.method()
        | [Colon _] -> PyCall(PyAttr(pyBase, "ravel"), [], [])
        | _ ->
            let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
            PyIndex(pyBase, pyIndices)

/// Translate an expression on the LHS of an assignment.
/// In MATLAB, LHS expressions are always indexing (never function calls),
/// and A(:) = v means "assign to all elements" (Python: A[:] = v), not ravel.
and private translateLhsExpr (expr: Expr) (tctx: TranslateContext) : PyExpr =
    match expr with
    | Apply(_, base_, args) ->
        let pyBase = translateLhsExpr base_ tctx
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        PyIndex(pyBase, pyIndices)
    | CurlyApply(_, base_, args) ->
        let pyBase = translateLhsExpr base_ tctx
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        PyIndex(pyBase, pyIndices)
    | FieldAccess(_, base_, field) ->
        if field = "<dynamic>" then
            PyCall(PyVar "getattr", [translateLhsExpr base_ tctx; PyStr "<dynamic>"], [])
        else
            PyAttr(translateLhsExpr base_ tctx, safeName field)
    | Var(_, name) -> PyVar (safeName name)
    | _ -> translateExpr expr tctx

and private translateBuiltinCall (mapping: BuiltinMapping) (fname: string) (args: IndexArg list) (tctx: TranslateContext) : PyExpr =
    let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
    let kwargs = if mapping.needsOrderF then [("order", PyStr "F")] else []

    // Track extra imports needed by certain builtins
    if mapping.pythonFunc.StartsWith("warnings.") then
        tctx.usedImports <- Set.add "warnings" tctx.usedImports
    if mapping.pythonFunc.StartsWith("plt.") then
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
    if mapping.pythonFunc.StartsWith("os.") then
        tctx.usedImports <- Set.add "os" tctx.usedImports
    if mapping.pythonFunc.StartsWith("scipy.") then
        tctx.usedImports <- Set.add "scipy" tctx.usedImports

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
        elif pyArgs.Length = 1 then
            // zeros(n) -> np.zeros((n, n))  (MATLAB: n-by-n matrix)
            let dim = pyArgs.[0]
            PyCall(PyVar mapping.pythonFunc, [PyTuple [dim; dim]], kwargs)
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
        | [obj; dim] -> PyIndex(PyAttr(obj, "shape"), [PyScalarIdx(mkBinOp "-" dim (PyConst 1.0))])
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
    | MinMaxDispatch elemwiseFunc ->
        // max(A) -> np.max(A);  max(A, B) -> np.maximum(A, B)
        // max(A, [], dim) -> np.max(A, axis=dim-1)
        match pyArgs with
        | [_] -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
        | [a; _; dim] when (match args with _ :: IndexExpr(_, MatrixLit(_, [])) :: _ -> true | _ -> false) ->
            PyCall(PyVar mapping.pythonFunc, [a], [("axis", mkBinOp "-" dim (PyConst 1.0))])
        | [a; b] -> PyCall(PyVar elemwiseFunc, [a; b], [])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
    | FlatNonzero ->
        // find(A) -> np.flatnonzero(A) + 1  (MATLAB returns 1-based indices)
        // find(A, n) -> (np.flatnonzero(A) + 1)[:n]  (first n results)
        match pyArgs with
        | [obj] -> mkBinOp "+" (PyCall(PyVar "np.flatnonzero", [obj], [])) (PyConst 1.0)
        | [obj; n] -> PyIndex(mkBinOp "+" (PyCall(PyVar "np.flatnonzero", [obj], [])) (PyConst 1.0), [PySlice(None, Some n, None)])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
    | WithKwarg(key, value) ->
        // std(A) -> np.std(A, ddof=1)
        let extraKwargs = [(key, PyConst value)]
        PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs @ extraKwargs)
    | StrjoinStyle ->
        // strjoin(parts) -> ''.join(parts); strjoin(parts, sep) -> sep.join(parts)
        match pyArgs with
        | [parts] -> PyCall(PyAttr(PyStr "", "join"), [parts], [])
        | [parts; sep] -> PyCall(PyAttr(sep, "join"), [parts], [])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | SortStyle ->
        // sort(A) -> np.sort(A) for single return; multi-return handled in AssignMulti
        PyCall(PyVar "np.sort", pyArgs, kwargs)
    | BinOpStyle op ->
        // strcmp(a, b) -> a == b
        match pyArgs with
        | [a; b] -> PyBinOp(op, a, b)
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | IsEmptyStyle ->
        // isempty(A) -> A.size == 0
        match pyArgs with
        | [obj] -> PyBinOp("==", PyAttr(obj, "size"), PyConst 0.0)
        | _ -> PyCall(PyVar "np.size", pyArgs, [])
    | RaiseStyle ->
        // error(msg) -> raise ValueError(msg)  — emit as PyRaise
        // We return a PyCall here; the caller in translateStmt will
        // wrap ExprStmt calls, but error() is special.
        // We handle it by emitting a raise expression wrapper.
        PyCall(PyVar ("raise " + mapping.pythonFunc), pyArgs, [])
    | SprintfStyle ->
        // sprintf(fmt, a, b) -> fmt % (a, b)
        match pyArgs with
        | [fmt] -> fmt  // no format args: just the string itself
        | fmt :: fmtArgs -> PyBinOp("%", fmt, PyTuple fmtArgs)
        | _ -> PyCall(PyVar "str", pyArgs, [])
    | FprintfStyle ->
        // fprintf(fmt, a, b) -> print(fmt % (a, b))
        // fprintf(fid, fmt, a, b) -> print(fmt % (a, b))  (drop file handle)
        match pyArgs with
        | [fmt] -> PyCall(PyVar "print", [fmt], [])
        | fmt :: fmtArgs ->
            // Check if first arg looks like a file handle (numeric constant)
            let actualFmt, actualArgs =
                match fmt with
                | PyConst _ when fmtArgs.Length >= 1 ->
                    // First arg is numeric (file handle), skip it
                    fmtArgs.Head, fmtArgs.Tail
                | _ -> fmt, fmtArgs
            if actualArgs.IsEmpty then
                PyCall(PyVar "print", [actualFmt], [])
            else
                PyCall(PyVar "print", [PyBinOp("%", actualFmt, PyTuple actualArgs)], [])
        | _ -> PyCall(PyVar "print", pyArgs, [])
    | MethodStyle methodName ->
        // lower(s) -> s.lower();  upper(s) -> s.upper()
        match pyArgs with
        | [obj] -> PyCall(PyAttr(obj, methodName), [], [])
        | obj :: rest -> PyCall(PyAttr(obj, methodName), rest, [])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
    | StrcmpiStyle ->
        // strcmpi(a, b) -> a.lower() == b.lower()
        match pyArgs with
        | [a; b] -> PyBinOp("==", PyCall(PyAttr(a, "lower"), [], []), PyCall(PyAttr(b, "lower"), [], []))
        | _ -> PyCall(PyVar "str.__eq__", pyArgs, [])
    | DimArgStyle ->
        // sum(A) -> np.sum(A);  sum(A, dim) -> np.sum(A, axis=dim-1)
        match pyArgs with
        | [a] -> PyCall(PyVar mapping.pythonFunc, [a], kwargs)
        | [a; dim] -> PyCall(PyVar mapping.pythonFunc, [a], kwargs @ [("axis", mkBinOp "-" dim (PyConst 1.0))])
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, kwargs)
    | CellfunStyle ->
        // cellfun(@func, C) -> list(map(func, C))
        // arrayfun(@func, A) -> list(map(func, A))
        match args with
        | IndexExpr(_, FuncHandle(_, funcName)) :: rest when rest.Length >= 1 ->
            let pyArrays = rest |> List.map (fun a -> translateCallArg a tctx)
            PyCall(PyVar "list", [PyCall(PyVar "map", PyVar funcName :: pyArrays, [])], [])
        | _ ->
            match pyArgs with
            | func :: arrays when arrays.Length >= 1 ->
                PyCall(PyVar "list", [PyCall(PyVar "map", func :: arrays, [])], [])
            | _ -> PyCall(PyVar "cellfun", pyArgs, [])
    | ClassStyle ->
        // class(x) -> type(x).__name__
        match pyArgs with
        | [arg] -> PyAttr(PyCall(PyVar "type", [arg], []), "__name__")
        | _ -> PyCall(PyVar "type", pyArgs, [])
    | IsTypeStyle typeName ->
        // ischar(x) -> isinstance(x, str)
        match pyArgs with
        | [arg] -> PyCall(PyVar "isinstance", [arg; PyVar typeName], [])
        | _ -> PyCall(PyVar "isinstance", pyArgs, [])
    | StructStyle ->
        // struct() -> SimpleNamespace()
        // struct('a', 1, 'b', 2) -> SimpleNamespace(a=1, b=2)
        tctx.usedImports <- Set.add "types" tctx.usedImports
        match args with
        | [] -> PyCall(PyVar "types.SimpleNamespace", [], [])
        | _ ->
            // Pair up: string key, value, string key, value, ...
            let rec pairUp (lst: IndexArg list) acc =
                match lst with
                | IndexExpr(_, StringLit(_, key)) :: IndexExpr(_, valExpr) :: rest ->
                    pairUp rest ((key, translateExpr valExpr tctx) :: acc)
                | _ -> List.rev acc
            let kwargs = pairUp args []
            if kwargs.IsEmpty then
                PyCall(PyVar "types.SimpleNamespace", pyArgs, [])
            else
                PyCall(PyVar "types.SimpleNamespace", [], kwargs)
    | ExistStyle ->
        // exist('x', 'var') -> x is not None  (variable existence)
        // exist(path, 'file') -> os.path.exists(path)  (file existence)
        tctx.usedImports <- Set.add "os" tctx.usedImports
        match args with
        | [IndexExpr(_, StringLit(_, varName)); IndexExpr(_, StringLit(_, "var"))] ->
            // exist('x', 'var') -> x is not None  (safeName guards against Python keywords)
            PyBinOp("is not", PyVar (safeName varName), PyNone)
        | [pathArg; IndexExpr(_, StringLit(_, "file"))] ->
            // exist(path, 'file') -> os.path.exists(path)
            let pyPath = translateCallArg pathArg tctx
            PyCall(PyVar "os.path.exists", [pyPath], [])
        | [pathArg; IndexExpr(_, StringLit(_, "dir"))] ->
            // exist(path, 'dir') -> os.path.isdir(path)
            let pyPath = translateCallArg pathArg tctx
            PyCall(PyVar "os.path.isdir", [pyPath], [])
        | [pathArg] ->
            // exist(path) -> os.path.exists(path)
            let pyPath = translateCallArg pathArg tctx
            PyCall(PyVar "os.path.exists", [pyPath], [])
        | _ -> PyCall(PyVar "os.path.exists", pyArgs, [])
    | FevalStyle ->
        // feval(f, a1, a2, ...) -> f(a1, a2, ...)
        // feval('sin', x) -> sin(x)  (string func name -> PyVar)
        // feval(@cos, x) -> cos(x)   (handle already translated to PyVar)
        match pyArgs with
        | PyStr funcName :: fArgs -> PyCall(PyVar funcName, fArgs, [])
        | func :: fArgs -> PyCall(func, fArgs, [])
        | _ -> PyCall(PyVar "feval", pyArgs, [])
    | RegexpStyle ->
        // regexp(str, pat) -> re.search(pat, str)
        // regexp(str, pat, 'match') -> re.findall(pat, str)
        // regexp(str, pat, 'tokens') -> re.findall(pat, str)
        tctx.usedImports <- Set.add "re" tctx.usedImports
        match args with
        | [IndexExpr(_, _); IndexExpr(_, _); IndexExpr(_, StringLit(_, opt))] ->
            match opt with
            | "match" | "tokens" ->
                let pyStr = pyArgs.[0]
                let pyPat = pyArgs.[1]
                PyCall(PyVar "re.findall", [pyPat; pyStr], [])
            | "split" ->
                let pyStr = pyArgs.[0]
                let pyPat = pyArgs.[1]
                PyCall(PyVar "re.split", [pyPat; pyStr], [])
            | _ ->
                match pyArgs with
                | [s; p] -> PyCall(PyVar "re.search", [p; s], [])
                | _ -> PyCall(PyVar "re.search", pyArgs, [])
        | _ ->
            match pyArgs with
            | [s; p] -> PyCall(PyVar "re.search", [p; s], [])
            | _ -> PyCall(PyVar "re.search", pyArgs, [])
    | RegexpRepStyle ->
        // regexprep(str, pat, rep) -> re.sub(pat, rep, str)
        tctx.usedImports <- Set.add "re" tctx.usedImports
        match pyArgs with
        | [s; p; r] -> PyCall(PyVar "re.sub", [p; r; s], [])
        | _ -> PyCall(PyVar "re.sub", pyArgs, [])
    | NanFullStyle ->
        // nan(m, n) -> np.full((m, n), np.nan)
        // nan(n) -> np.full((n, n), np.nan)  (MATLAB: square matrix)
        match pyArgs with
        | [] -> PyAttr(PyVar "np", "nan")  // just 'nan' constant
        | [n] -> PyCall(PyVar "np.full", [PyTuple [n; n]; PyAttr(PyVar "np", "nan")], [])
        | dims -> PyCall(PyVar "np.full", [PyTuple dims; PyAttr(PyVar "np", "nan")], [])
    | FileIoStyle methodName ->
        // fwrite(fd, data) -> fd.write(data) with special handling for fd=1 (stdout) and fd=2 (stderr)
        tctx.usedImports <- Set.add "sys" tctx.usedImports
        let receiver =
            match pyArgs with
            | PyConst 1.0 :: _ -> PyAttr(PyVar "sys", "stdout")
            | PyConst 2.0 :: _ -> PyAttr(PyVar "sys", "stderr")
            | obj :: _ -> obj
            | [] -> PyVar "fd"
        let restArgs = if pyArgs.Length > 1 then pyArgs.[1..] else []
        PyCall(PyAttr(receiver, methodName), restArgs, [])
    | CatStyle ->
        // cat(dim, A, B, ...) -> np.concatenate((A, B, ...), axis=dim-1)
        match pyArgs with
        | dim :: arrays when arrays.Length >= 1 ->
            PyCall(PyVar "np.concatenate", [PyTuple arrays], [("axis", mkBinOp "-" dim (PyConst 1.0))])
        | _ -> PyCall(PyVar "np.concatenate", pyArgs, [])
    | Sub2IndStyle ->
        // sub2ind(sz, i1, i2, ...) -> np.ravel_multi_index((i1-1, i2-1, ...), sz, order='F') + 1
        // The +1 keeps the result 1-based so downstream MATLAB-style indexing (which subtracts 1) works correctly.
        match pyArgs with
        | sz :: subscripts when subscripts.Length >= 1 ->
            let shifted = subscripts |> List.map (fun s -> mkBinOp "-" s (PyConst 1.0))
            mkBinOp "+" (PyCall(PyVar mapping.pythonFunc, [PyTuple shifted; sz], [("order", PyStr "F")])) (PyConst 1.0)
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
    | Ind2SubStyle ->
        // ind2sub(sz, ind) -> np.unravel_index(ind-1, sz, order='F') + 1
        // The +1 keeps subscripts 1-based for downstream MATLAB-style indexing.
        match pyArgs with
        | [sz; ind] ->
            mkBinOp "+" (PyCall(PyVar mapping.pythonFunc, [mkBinOp "-" ind (PyConst 1.0); sz], [("order", PyStr "F")])) (PyConst 1.0)
        | _ -> PyCall(PyVar mapping.pythonFunc, pyArgs, [])
    | PermuteStyle ->
        // permute(A, [2 1 3]) -> np.transpose(A, (1, 0, 2))
        // Subtract 1 from each dimension index (MATLAB 1-based -> Python 0-based)
        match args with
        | [IndexExpr(_, arr); IndexExpr(_, MatrixLit(_, rows))] ->
            let pyArr = translateExpr arr tctx
            let dims =
                rows |> List.collect id
                |> List.map (fun e ->
                    match e with
                    | Const(_, v) -> PyConst(v - 1.0)
                    | _ -> mkBinOp "-" (translateExpr e tctx) (PyConst 1.0))
            PyCall(PyVar "np.transpose", [pyArr; PyTuple dims], [])
        | _ ->
            // Non-literal order: subtract 1 from each element at runtime
            match pyArgs with
            | [arr; order] -> PyCall(PyVar "np.transpose", [arr; mkBinOp "-" order (PyConst 1.0)], [])
            | _ -> PyCall(PyVar "np.transpose", pyArgs, [])
    | BsxfunStyle ->
        // bsxfun(@op, A, B) -> A op B  (NumPy broadcasts natively)
        // The first arg is a FuncHandle which translateCallArg turns into PyVar.
        // We need to look at the raw IR arg to get the function name.
        match args with
        | IndexExpr(_, FuncHandle(_, opName)) :: rest when rest.Length >= 2 ->
            let pyA = translateCallArg rest.[0] tctx
            let pyB = translateCallArg rest.[1] tctx
            match opName with
            | "times" | "mtimes" -> PyBinOp("*", pyA, pyB)
            | "plus"   -> PyBinOp("+", pyA, pyB)
            | "minus"  -> PyBinOp("-", pyA, pyB)
            | "rdivide" -> PyBinOp("/", pyA, pyB)
            | "ldivide" -> PyBinOp("/", pyB, pyA)  // a.\b = b./a
            | "power"  -> PyBinOp("**", pyA, pyB)
            | "eq"     -> PyBinOp("==", pyA, pyB)
            | "ne"     -> PyBinOp("!=", pyA, pyB)
            | "lt"     -> PyBinOp("<", pyA, pyB)
            | "le"     -> PyBinOp("<=", pyA, pyB)
            | "gt"     -> PyBinOp(">", pyA, pyB)
            | "ge"     -> PyBinOp(">=", pyA, pyB)
            | "and"    -> PyBinOp("&", pyA, pyB)
            | "or"     -> PyBinOp("|", pyA, pyB)
            | _ ->
                // Unknown handle: call it as a function
                PyCall(PyVar opName, [pyA; pyB], [])
        | _ ->
            // Fallback: pass through as-is
            PyCall(PyVar "bsxfun", pyArgs, [])

and private translateCallArg (arg: IndexArg) (tctx: TranslateContext) : PyExpr =
    match arg with
    | IndexExpr(_, expr) -> translateExpr expr tctx
    | Colon _ -> PyStr ":"  // rare in function calls
    | Ir.Range(_, start, end_) ->
        PyCall(PyVar "range", [translateExpr start tctx; mkBinOp "+" (translateExpr end_ tctx) (PyConst 1.0)], [])
    | Ir.SteppedRange(_, start, step, end_) ->
        PyCall(PyVar "range", [translateExpr start tctx; mkBinOp "+" (translateExpr end_ tctx) (PyConst 1.0); translateExpr step tctx], [])

and private containsEndExpr (expr: Expr) : bool =
    match expr with
    | End _ -> true
    | Neg(_, operand)
    | Not(_, operand)
    | Transpose(_, operand) -> containsEndExpr operand
    | BinOp(_, _, left, right) -> containsEndExpr left || containsEndExpr right
    | FieldAccess(_, base_, _) -> containsEndExpr base_
    | Lambda(_, _, body) -> containsEndExpr body
    | Apply(_, base_, args)
    | CurlyApply(_, base_, args) ->
        containsEndExpr base_ || (args |> List.exists containsEndIndexArg)
    | MatrixLit(_, rows)
    | CellLit(_, rows) ->
        rows |> List.exists (List.exists containsEndExpr)
    | _ -> false

and private containsEndIndexArg (arg: IndexArg) : bool =
    match arg with
    | IndexExpr(_, expr) -> containsEndExpr expr
    | Colon _ -> false
    | Ir.Range(_, start, end_) -> containsEndExpr start || containsEndExpr end_
    | Ir.SteppedRange(_, start, step, end_) -> containsEndExpr start || containsEndExpr step || containsEndExpr end_

and private translateIndexArg (arg: IndexArg) (tctx: TranslateContext) : PyIdx =
    match arg with
    | Colon _ -> PySlice(None, None, None)  // : -> :
    | IndexExpr(_, End _) ->
        // A(end) -> A[-1]  (Python -1 = last element, no offset needed)
        PyScalarIdx(PyConst -1.0)
    | IndexExpr(_, expr) ->
        if containsEndExpr expr then
            // A(end-k) -> A[-1-k] (already end-relative; no extra 0-based shift)
            PyScalarIdx(translateExpr expr tctx)
        else
            // A(i) -> A[i-1]
            PyScalarIdx(mkBinOp "-" (translateExpr expr tctx) (PyConst 1.0))
    | Ir.Range(_, start, end_) ->
        // A(a:b) -> A[a-1:b]  (0-based start, exclusive end cancels)
        // A(a:end) -> A[a-1:] (None = go to end)
        let pyStart = mkBinOp "-" (translateExpr start tctx) (PyConst 1.0)
        let pyEnd = match end_ with End _ -> None | _ -> Some (translateExpr end_ tctx)
        PySlice(Some pyStart, pyEnd, None)
    | Ir.SteppedRange(_, start, step, end_) ->
        let pyStart = mkBinOp "-" (translateExpr start tctx) (PyConst 1.0)
        let pyEnd = match end_ with End _ -> None | _ -> Some (translateExpr end_ tctx)
        let pyStep = translateExpr step tctx
        PySlice(Some pyStart, pyEnd, Some pyStep)

and private translateMatrixLit (rows: Expr list list) (tctx: TranslateContext) : PyExpr =
    let pyRows = rows |> List.map (fun row -> row |> List.map (fun e -> translateExpr e tctx))
    match pyRows with
    | [[single]] -> single  // [x] -> x (scalar)
    | [row] -> PyCall(PyVar "np.array", [PyList row], [])  // [a, b, c] -> np.array([a, b, c])
    | [] -> PyCall(PyVar "np.array", [PyList []], [])  // [] -> np.array([])
    | _ -> PyCall(PyVar "np.array", [PyList (pyRows |> List.map PyList)], [])

/// Check if an expression references a variable name
let rec private exprReferencesVar (name: string) (expr: Expr) : bool =
    match expr with
    | Var(_, n) -> n = name
    | Neg(_, e) | Not(_, e) | Transpose(_, e) -> exprReferencesVar name e
    | BinOp(_, _, l, r) -> exprReferencesVar name l || exprReferencesVar name r
    | FieldAccess(_, b, _) -> exprReferencesVar name b
    | Lambda(_, _, body) -> exprReferencesVar name body
    | Apply(_, b, args) | CurlyApply(_, b, args) ->
        exprReferencesVar name b || args |> List.exists (indexArgReferencesVar name)
    | MatrixLit(_, rows) | CellLit(_, rows) -> rows |> List.exists (List.exists (exprReferencesVar name))
    | _ -> false

and private indexArgReferencesVar (name: string) (arg: IndexArg) : bool =
    match arg with
    | IndexExpr(_, e) -> exprReferencesVar name e
    | Colon _ -> false
    | Ir.Range(_, s, e) -> exprReferencesVar name s || exprReferencesVar name e
    | Ir.SteppedRange(_, s, st, e) -> exprReferencesVar name s || exprReferencesVar name st || exprReferencesVar name e

/// Check if a statement (or its children) references a variable name
and private stmtReferencesVar (name: string) (stmt: Stmt) : bool =
    match stmt with
    | Assign(_, _, e) | ExprStmt(_, e) -> exprReferencesVar name e
    | AssignMulti(_, _, e) -> exprReferencesVar name e
    | If(_, c, t, el) -> exprReferencesVar name c || t |> List.exists (stmtReferencesVar name) || el |> List.exists (stmtReferencesVar name)
    | IfChain(_, cs, bs, el) -> cs |> List.exists (exprReferencesVar name) || bs |> List.exists (List.exists (stmtReferencesVar name)) || el |> List.exists (stmtReferencesVar name)
    | While(_, c, b) -> exprReferencesVar name c || b |> List.exists (stmtReferencesVar name)
    | For(_, _, it, b) -> exprReferencesVar name it || b |> List.exists (stmtReferencesVar name)
    | Switch(_, e, cases, ow) -> exprReferencesVar name e || cases |> List.exists (fun (v,b) -> exprReferencesVar name v || b |> List.exists (stmtReferencesVar name)) || ow |> List.exists (stmtReferencesVar name)
    | Try(_, t, c) -> t |> List.exists (stmtReferencesVar name) || c |> List.exists (stmtReferencesVar name)
    | IndexAssign(_, _, args, e) -> args |> List.exists (indexArgReferencesVar name) || exprReferencesVar name e
    | LhsAssign(_, _, lhs, e) -> exprReferencesVar name lhs || exprReferencesVar name e
    | FunctionDef(_, _, _, _, b, _) -> b |> List.exists (stmtReferencesVar name)
    | _ -> false

// -------------------------------------------------------------------------
// Command-style call translation (for OpaqueStmt raw text)
// -------------------------------------------------------------------------

/// Try to translate common MATLAB command-style calls (hold on, axis equal, etc.)
/// Returns Some [stmts] on success, None if unrecognized.
let private translateCommandStyle (raw: string) (tctx: TranslateContext) : PyStmt list option =
    let words = raw.Split([|' '; '\t'|], System.StringSplitOptions.RemoveEmptyEntries) |> Array.toList
    match words with
    // hold on/off → noop (matplotlib holds by default)
    | ["hold"; "on"] | ["hold"; "off"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyCommentStmt (sprintf "%s (matplotlib default)" raw)]
    // axis commands → plt.axis(...)
    | "axis" :: args when args.Length >= 1 ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        let argStr = args |> List.filter (fun a -> a <> "off" || args = ["off"]) |> String.concat " "
        match args with
        | ["off"] -> Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "axis"), [PyStr "off"], []))]
        | ["equal"] | ["image"] | ["tight"] | ["square"] | ["manual"] | ["normal"] ->
            Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "axis"), [PyStr args.Head], []))]
        | _ ->
            // Multi-word axis args: axis equal off tight → plt.axis('equal')  (best-effort)
            let primary = args |> List.tryFind (fun a -> a <> "off" && a <> "on")
            match primary with
            | Some p -> Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "axis"), [PyStr p], []))]
            | None -> Some [PyCommentStmt (sprintf "MATLAB: %s" raw)]
    // grid on/off/minor → plt.grid(...)
    | ["grid"; "on"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "grid"), [PyBool true], []))]
    | ["grid"; "off"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "grid"), [PyBool false], []))]
    | ["grid"; "minor"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "grid"), [PyBool true], [("which", PyStr "minor")]))]
    // close all → plt.close('all')
    | ["close"; "all"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "close"), [PyStr "all"], []))]
    | ["close"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "close"), [], []))]
    // colormap X → plt.set_cmap('X')
    | ["colormap"; cmap] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "set_cmap"), [PyStr cmap], []))]
    // warning on/off → warnings.filterwarnings(...)
    | ["warning"; "on"] ->
        tctx.usedImports <- Set.add "warnings" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "warnings", "filterwarnings"), [PyStr "default"], []))]
    | ["warning"; "off"] ->
        tctx.usedImports <- Set.add "warnings" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "warnings", "filterwarnings"), [PyStr "ignore"], []))]
    // warning off/on <ID> → comment (MATLAB warning IDs have no Python equivalent)
    | "warning" :: "off" :: rest when rest.Length >= 1 ->
        Some [PyCommentStmt (sprintf "warning off %s" (rest |> String.concat " "))]
    | "warning" :: "on" :: rest when rest.Length >= 1 ->
        Some [PyCommentStmt (sprintf "warning on %s" (rest |> String.concat " "))]
    // load file → comment (workspace load has no Python equivalent)
    | "load" :: rest when rest.Length >= 1 ->
        Some [PyCommentStmt (sprintf "load %s" (rest |> String.concat " "))]
    // import pkg.* → comment (Java/MATLAB package imports have no Python equivalent)
    | "import" :: rest when rest.Length >= 1 ->
        Some [PyCommentStmt (sprintf "import %s" (rest |> String.concat " "))]
    // graphics toggle commands → comment
    | "rotate3d" :: rest ->
        Some [PyCommentStmt (sprintf "rotate3d %s" (rest |> String.concat " "))]
    | "shading" :: rest ->
        Some [PyCommentStmt (sprintf "shading %s" (rest |> String.concat " "))]
    | "lighting" :: rest ->
        Some [PyCommentStmt (sprintf "lighting %s" (rest |> String.concat " "))]
    // drawnow → plt.draw(); plt.pause(0.001)
    | ["drawnow"] ->
        tctx.usedImports <- Set.add "matplotlib" tctx.usedImports
        Some [PyExprStmt(PyCall(PyAttr(PyVar "plt", "draw"), [], []))
              PyExprStmt(PyCall(PyAttr(PyVar "plt", "pause"), [PyConst 0.001], []))]
    // clear / clear all / clear classes / clear functions / clear global → comment
    | ["clear"] | ["clear"; "all"] | ["clear"; "classes"] | ["clear"; "functions"] | ["clear"; "global"] ->
        Some [PyCommentStmt "clear all variables"]
    // clear var1 var2 ... → comment listing the variables
    | "clear" :: vars when vars.Length >= 1 ->
        Some [PyCommentStmt (sprintf "clear %s" (vars |> String.concat " "))]
    | _ -> None

// -------------------------------------------------------------------------
// Statement translation
// -------------------------------------------------------------------------

let rec translateStmt (stmt: Stmt) (tctx: TranslateContext) : PyStmt list =
    // Filter out synthetic parser sentinels
    if isSyntheticSentinel stmt then [] else
    match stmt with
    | Assign(_, name, Var(srcLoc, srcName)) when Set.contains srcLoc tctx.copySites ->
        // Copy semantics: B = A -> B = A.copy()
        [PyAssign(safeName name, PyCall(PyAttr(PyVar (safeName srcName), "copy"), [], []))]
    | Assign(_, name, expr) ->
        [PyAssign(safeName name, translateExpr expr tctx)]
    | AssignMulti(_, targets, expr) ->
        // Handle special multi-return builtins
        match expr with
        | Apply(_, Var(_, "sort"), args) when targets.Length = 2 ->
            // [sorted, idx] = sort(A) -> sorted = np.sort(A); idx = np.argsort(A) + 1
            let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
            let t0 = if targets.[0] = "~" then "_" else safeName targets.[0]
            let t1 = if targets.[1] = "~" then "_" else safeName targets.[1]
            let sortCall = PyCall(PyVar "np.sort", pyArgs, [])
            let argsortCall = PyBinOp("+", PyCall(PyVar "np.argsort", pyArgs, []), PyConst 1.0)
            [PyAssign(t0, sortCall); PyAssign(t1, argsortCall)]
        | _ ->
        // Handle return-order swaps for builtins with different conventions
        let swappedTargets =
            match expr with
            | Apply(_, Var(_, "eig"), _) when targets.Length = 2 ->
                // MATLAB: [V, D] = eig(A)  -> eigenvalues first in NumPy
                [targets.[1]; targets.[0]]
            | Apply(_, Var(_, "svd"), _) when targets.Length = 3 ->
                // MATLAB: [U, S, V] = svd(A)  -> NumPy returns (U, S, Vh) — same order
                targets
            | _ -> targets
        // MATLAB ~ (ignore output) -> Python _ convention
        let pyTargets = swappedTargets |> List.map (fun t -> if t = "~" then "_" else safeName t)
        [PyMultiAssign(pyTargets, translateExpr expr tctx)]
    | ExprStmt(_, Var(_, "clear")) ->
        // Bare 'clear' parses as Var, not OpaqueStmt; translate to comment
        [PyCommentStmt "clear all variables"]
    | ExprStmt(_, Var(_, "load")) ->
        // Bare 'load' parses as Var; no Python equivalent
        [PyCommentStmt "load (no file specified)"]
    | ExprStmt(_, Var(_, "import")) ->
        // Bare 'import' parses as Var; no Python equivalent
        [PyCommentStmt "import (no package specified)"]
    | ExprStmt(_, Apply(_, Var(_, "addpath"), args)) ->
        tctx.usedImports <- Set.add "sys" tctx.usedImports
        let paths =
            args |> List.choose (fun arg ->
                match arg with
                | IndexExpr(_, StringLit(_, s)) -> Some s
                | IndexExpr(_, Apply(_, Var(_, "genpath"), [IndexExpr(_, StringLit(_, s))])) -> Some s
                | IndexExpr(_, Apply(_, Var(_, "fullfile"), innerArgs)) ->
                    let strs = innerArgs |> List.choose (fun a -> match a with IndexExpr(_, StringLit(_, s)) -> Some s | _ -> None)
                    if strs.Length = innerArgs.Length then Some (String.concat "/" strs) else None
                | _ -> None)
        tctx.addpathDirs <- tctx.addpathDirs @ paths
        if paths.IsEmpty then
            [PyCommentStmt (sprintf "MATLAB: addpath (dynamic args)")]
        else
            paths |> List.map (fun p ->
                PyExprStmt(PyCall(PyAttr(PyAttr(PyVar "sys", "path"), "insert"), [PyConst 0.0; PyStr p], [])))

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
        let safeVar = safeName var_
        let pyBody = body |> List.collect (fun s -> translateStmt s tctx)
        match it with
        | BinOp(_, ":", BinOp(_, ":", start, step), end_) ->
            // for i = a:step:b -> for i in range(a, b + 1, step):
            let pyStart = translateExpr start tctx
            let pyEnd = mkBinOp "+" (translateExpr end_ tctx) (PyConst 1.0)
            let pyStep = translateExpr step tctx
            [PyFor(safeVar, PyCall(PyVar "range", [pyStart; pyEnd; pyStep], []), pyBody)]
        | BinOp(_, ":", start, Neg(_, BinOp(_, ":", stepAbs, end_))) ->
            // Parser artifact: 10:-1:1 parses as BinOp(":", 10, Neg(BinOp(":", 1, 1)))
            // Reinterpret as: range(start, end - 1, -stepAbs)
            let pyStart = translateExpr start tctx
            let pyEnd = mkBinOp "-" (translateExpr end_ tctx) (PyConst 1.0)
            let pyStep = PyUnaryOp("-", translateExpr stepAbs tctx)
            [PyFor(safeVar, PyCall(PyVar "range", [pyStart; pyEnd; pyStep], []), pyBody)]
        | BinOp(_, ":", start, end_) ->
            // for i = a:b -> for i in range(a, b + 1):
            let pyStart = translateExpr start tctx
            let pyEnd = mkBinOp "+" (translateExpr end_ tctx) (PyConst 1.0)
            [PyFor(safeVar, PyCall(PyVar "range", [pyStart; pyEnd], []), pyBody)]
        | Apply(_, Var(_, "colon"), _) ->
            // Fallback for colon() calls
            [PyCommentStmt (sprintf "CONFORMAL: complex for-loop iterator for '%s'" safeVar)]
             @ [PyFor(safeVar, translateExpr it tctx, pyBody)]
        | _ ->
            // General iterator expression (could be Range IR node)
            let pyIt = translateForIterator it tctx
            [PyFor(safeVar, pyIt, pyBody)]

    | Switch(_, expr, cases, otherwise) ->
        let pyExpr = translateExpr expr tctx
        let makeCaseCond (caseVal: Expr) =
            match caseVal with
            | CellLit(_, rows) ->
                // case {1, 2, 3} -> x in (1, 2, 3)  (match any of these values)
                let allExprs = rows |> List.collect id |> List.map (fun e -> translateExpr e tctx)
                PyBinOp("in", pyExpr, PyTuple allExprs)
            | _ -> PyBinOp("==", pyExpr, translateExpr caseVal tctx)
        match cases with
        | [] -> []
        | (firstVal, firstBody) :: restCases ->
            let pyFirstCond = makeCaseCond firstVal
            let pyFirstBody = firstBody |> List.collect (fun s -> translateStmt s tctx)
            let elifs =
                restCases |> List.map (fun (v, b) ->
                    (makeCaseCond v,
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
        [PyExprStmt(PyBinOp("=", PyIndex(PyVar (safeName baseName), pyIndices), pyExpr))]

    | StructAssign(_, baseName, fields, expr) ->
        let pyExpr = translateExpr expr tctx
        let target = fields |> List.fold (fun acc f -> sprintf "%s.%s" acc (safeName f)) (safeName baseName)
        [PyAssign(target, pyExpr)]

    | CellAssign(_, baseName, args, expr) ->
        let pyIndices = args |> List.map (fun a -> translateIndexArg a tctx)
        let pyExpr = translateExpr expr tctx
        [PyExprStmt(PyBinOp("=", PyIndex(PyVar (safeName baseName), pyIndices), pyExpr))]

    | IndexStructAssign(_, baseName, indexArgs, _, fields, expr) ->
        // base(i, j).field1.field2 = expr -> base[i-1, j-1].field1.field2 = expr
        let pyExpr = translateExpr expr tctx
        let pyIndices = indexArgs |> List.map (fun a -> translateIndexArg a tctx)
        let indexedBase = PyIndex(PyVar (safeName baseName), pyIndices)
        let target = fields |> List.fold (fun acc f -> PyAttr(acc, safeName f)) indexedBase
        [PyExprStmt(PyBinOp("=", target, pyExpr))]

    | FieldIndexAssign(_, baseName, prefixFields, indexArgs, _, suffixFields, expr) ->
        // base.field1(i).field2.field3 = expr -> base.field1[i-1].field2.field3 = expr
        let pyExpr = translateExpr expr tctx
        let base_ = prefixFields |> List.fold (fun acc f -> PyAttr(acc, safeName f)) (PyVar (safeName baseName))
        let pyIndices = indexArgs |> List.map (fun a -> translateIndexArg a tctx)
        let indexedBase = PyIndex(base_, pyIndices)
        let target = suffixFields |> List.fold (fun acc f -> PyAttr(acc, safeName f)) indexedBase
        [PyExprStmt(PyBinOp("=", target, pyExpr))]

    | LhsAssign(_, _, lhsExpr, expr) ->
        // General chain assignment: net.layers{l}.a{j} = expr -> net.layers[l-1].a[j-1] = expr
        // Uses translateLhsExpr so Apply is always indexing (not function call) and A(:) = v -> A[:] = v
        let pyLhs = translateLhsExpr lhsExpr tctx
        let pyRhs = translateExpr expr tctx
        [PyExprStmt(PyBinOp("=", pyLhs, pyRhs))]

    | OpaqueStmt(_, targets, raw) ->
        let trimmed = raw.Trim()
        // Handle global/persistent declarations
        if trimmed.StartsWith("global ") || trimmed = "global" then
            let vars = targets |> List.map safeName
            if tctx.functionDepth > 1 then
                // Inside nested function: use nonlocal
                vars |> List.map (fun v -> PyExprStmt(PyVar (sprintf "nonlocal %s" v)))
            else
                // Top-level function: use global
                vars |> List.map (fun v -> PyExprStmt(PyVar (sprintf "global %s" v)))
        elif trimmed.StartsWith("persistent ") || trimmed = "persistent" then
            let varNames = targets |> String.concat " "
            [PyCommentStmt (sprintf "persistent %s (no Python equivalent; use module-level or function attributes)" varNames)]
        elif trimmed.StartsWith("addpath ") then
            tctx.usedImports <- Set.add "sys" tctx.usedImports
            let paths =
                trimmed.Substring(8).Split([|' '; '\t'|], System.StringSplitOptions.RemoveEmptyEntries)
                |> Array.toList
                |> List.map (fun s -> s.Trim('\'').Trim('"'))
            tctx.addpathDirs <- tctx.addpathDirs @ paths
            paths |> List.map (fun p ->
                PyExprStmt(PyCall(PyAttr(PyAttr(PyVar "sys", "path"), "insert"), [PyConst 0.0; PyStr p], [])))
        else
        // Try to translate common command-style calls before falling back
        match translateCommandStyle trimmed tctx with
        | Some stmts -> stmts
        | None ->
            let comment = [PyCommentStmt (sprintf "MATLAB: %s" (raw.Trim()))]
            match targets with
            | [] -> comment
            | _ ->
                (targets |> List.map (fun t -> PyAssign(safeName t, PyNone)))
                @ comment

    | FunctionDef(_, name, parms, outputVars, body, _) ->
        let savedReturnVars = tctx.currentReturnVars
        let savedDepth = tctx.functionDepth
        tctx.functionDepth <- tctx.functionDepth + 1
        let safeOutputVars = outputVars |> List.map safeName
        tctx.currentReturnVars <- safeOutputVars
        // Check for varargin: if last param is "varargin", replace with *args
        let hasVarargin = parms |> List.tryLast = Some "varargin"
        let baseParms = if hasVarargin then parms |> List.filter (fun p -> p <> "varargin") else parms
        // Check if body references 'nargin' / 'nargout' — add preamble as needed
        let usesNargin = body |> List.exists (fun s -> stmtReferencesVar "nargin" s)
        let usesNargout = body |> List.exists (fun s -> stmtReferencesVar "nargout" s)
        let safeParms = baseParms |> List.map safeName
        let pyParms =
            let ps =
                if hasVarargin then
                    // With *args, fixed params stay required; nargin computed from len(args) + fixed count
                    safeParms
                elif usesNargin && safeParms.Length > 0 then
                    safeParms |> List.map (fun p -> p + "=None")
                else safeParms
            if hasVarargin then ps @ ["*args"] else ps
        let narginPreamble =
            if usesNargin then
                if hasVarargin then
                    // nargin = len(args) + <number of fixed params>
                    let fixedCount = safeParms |> List.filter (fun p -> p <> "*args") |> List.length
                    if fixedCount > 0 then
                        [PyAssign("nargin", PyBinOp("+", PyCall(PyVar "len", [PyVar "args"], []), PyConst (float fixedCount)))]
                    else
                        [PyAssign("nargin", PyCall(PyVar "len", [PyVar "args"], []))]
                elif safeParms.Length > 0 then
                    let countExpr = sprintf "sum(1 for __x in [%s] if __x is not None)" (safeParms |> String.concat ", ")
                    [PyExprStmt(PyBinOp("=", PyVar "nargin", PyVar countExpr))]
                else []
            else []
        let nargoutPreamble =
            if usesNargout then
                // nargout = <number of declared output variables>
                // Always compute all outputs in Python (caller ignores extras with _)
                [PyAssign("nargout", PyConst (float safeOutputVars.Length))]
            else []
        let pyBody = narginPreamble @ nargoutPreamble @ (body |> List.collect (fun s -> translateStmt s tctx))
        // Add implicit return if the body doesn't end with an explicit return
        let needsReturn =
            not safeOutputVars.IsEmpty &&
            (pyBody.IsEmpty || (match List.last pyBody with PyReturn _ -> false | _ -> true))
        let fullBody =
            if needsReturn then
                pyBody @ [PyReturn (safeOutputVars |> List.map PyVar)]
            else
                pyBody
        tctx.currentReturnVars <- savedReturnVars
        tctx.functionDepth <- savedDepth
        [PyFuncDef(name, pyParms, (if fullBody.IsEmpty then [PyPass] else fullBody), safeOutputVars)]

and private translateForIterator (it: Expr) (tctx: TranslateContext) : PyExpr =
    match it with
    | Apply(_, Var(_, "colon"), args) ->
        // colon(a, b) or colon(a, step, b)
        let pyArgs = args |> List.map (fun a -> translateCallArg a tctx)
        match pyArgs with
        | [a; b] -> PyCall(PyVar "range", [a; mkBinOp "+" b (PyConst 1.0)], [])
        | [a; s; b] -> PyCall(PyVar "range", [a; mkBinOp "+" b (PyConst 1.0); s], [])
        | _ -> translateExpr it tctx
    | _ ->
        // General expression as iterator
        translateExpr it tctx

// -------------------------------------------------------------------------
// Program translation
// -------------------------------------------------------------------------

let translateProgram (program: Ir.Program) (tctx: TranslateContext) (sourceFile: string) : PyProgram =
    let body = program.body |> List.collect (fun s -> translateStmt s tctx)

    let extraImports =
        [ if Set.contains "warnings" tctx.usedImports then yield PyImport("warnings", None)
          if Set.contains "matplotlib" tctx.usedImports then yield PyFromImport("matplotlib", ["pyplot as plt"])
          if Set.contains "os" tctx.usedImports then yield PyImport("os", None)
          if Set.contains "scipy" tctx.usedImports then yield PyImport("scipy", None)
          if Set.contains "sys" tctx.usedImports then yield PyImport("sys", None)
          if Set.contains "types" tctx.usedImports then yield PyImport("types", None)
          if Set.contains "re" tctx.usedImports then yield PyImport("re", None) ]

    let baseImports = [
        PyCommentStmt (sprintf "Generated by conformal-migrate 3.5.0")
        PyCommentStmt (sprintf "Source: %s" sourceFile)
        PyImport("numpy", Some "np")
    ]
    let imports = baseImports @ extraImports

    { imports = imports; body = body }
