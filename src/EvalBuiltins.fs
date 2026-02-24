module EvalBuiltins

open Ir
open Shapes
open Env
open Context
open Diagnostics
open Builtins
open DimExtract
open Intervals
open SharedTypes

// ---------------------------------------------------------------------------
// EvalBuiltins: builtin function shape inference via dispatch table.
// Port of analysis/eval_builtins.py
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Declarative builtin sets (frozenset equivalents)
// ---------------------------------------------------------------------------

let PASSTHROUGH_BUILTINS : Set<string> =
    Set.ofList [
        "abs"; "sqrt"; "sin"; "cos"; "tan"; "asin"; "acos"; "atan"
        "tanh"; "cosh"; "sinh"; "atanh"; "acosh"; "asinh"; "conj"; "not"
        "flipud"; "fliplr"; "triu"; "tril"; "sort"; "unique"
        "exp"; "log"; "log2"; "log10"; "ceil"; "floor"; "round"; "sign"
        "real"; "imag"; "cumsum"; "cumprod"
        "expm"; "logm"; "sqrtm"; "circshift"; "null"; "orth"
        "sgolayfilt"; "squeeze"; "fftshift"; "ifftshift"; "unwrap"
        "deg2rad"; "rad2deg"; "angle"
        "quatconj"; "quatinv"; "quatnormalize"
    ]

let SCALAR_PREDICATE_BUILTINS : Set<string> =
    Set.ofList [
        "isscalar"; "iscell"; "isempty"; "isnumeric"; "islogical"; "ischar"
        "isnan"; "isinf"; "isfinite"; "issymmetric"; "isstruct"; "isreal"
        "issparse"; "isvector"; "isinteger"; "isfloat"; "isstring"; "issorted"
        "isfield"
    ]

let TYPE_CAST_BUILTINS : Set<string> =
    Set.ofList [
        "double"; "single"; "int8"; "int16"; "int32"; "int64"
        "uint8"; "uint16"; "uint32"; "uint64"; "logical"; "complex"; "typecast"
    ]

let REDUCTION_BUILTINS : Set<string> =
    Set.ofList [
        "sum"; "prod"; "mean"; "any"; "all"; "median"; "var"; "std"
        "nanmean"; "nansum"; "nanstd"; "nanmin"; "nanmax"
    ]

let SCALAR_QUERY_BUILTINS : Set<string> =
    Set.ofList [
        "length"; "numel"; "det"; "norm"; "trace"; "rank"; "cond"; "rcond"
        "nnz"; "sprank"; "str2double"; "dot"
        "quatnorm"
    ]

let MATRIX_CONSTRUCTOR_BUILTINS : Set<string> =
    Set.ofList [
        "eye"; "rand"; "randn"; "true"; "false"; "nan"; "NaN"; "inf"; "Inf"
    ]

let STRING_RETURN_BUILTINS : Set<string> =
    Set.ofList [
        "num2str"; "int2str"; "mat2str"; "char"; "string"; "sprintf"; "fullfile"
    ]

let SCALAR_NARY_BUILTINS : Set<string> =
    Set.ofList [ "strcmpi"; "strcmp"; "exist" ]


// ---------------------------------------------------------------------------
// Helper: unwrap an IndexArg to Expr (raises on Colon/Range without expr)
// ---------------------------------------------------------------------------

let private unwrapArg (arg: IndexArg) : Expr option =
    match arg with
    | IndexExpr(_, _, e) -> Some e
    | _ -> None


// ---------------------------------------------------------------------------
// Handler helpers
// ---------------------------------------------------------------------------

/// evalArgShape: evaluate an IndexArg to a Shape.
let private evalArgShape
    (arg: IndexArg)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape =
    match arg with
    | IndexExpr(_, _, e) -> evalExprFn e env warnings ctx
    | _ -> UnknownShape


/// checkNegativeDimArg: check a dimension argument for negative value, emit warning.
let private checkNegativeDimArg
    (arg: Expr)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (line: int)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : unit =
    let iv = getIntervalFn arg env ctx
    if intervalDefinitelyNegative (iv |> Option.map id) then
        let ivStr = match iv with Some i -> "[" + string i.lo + "," + string i.hi + "]" | None -> "?"
        warnings.Add(warnPossiblyNegativeDim line ivStr)


// ---------------------------------------------------------------------------
// Individual handler functions
// ---------------------------------------------------------------------------

let private handleZerosOnes
    (fname: string)
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape option =
    if args.Length = 1 then
        match unwrapArg args.[0] with
        | None -> None
        | Some arg ->
            checkNegativeDimArg arg env warnings ctx line getIntervalFn
            let d = exprToDimIrCtx arg env (Some ctx)
            Some (Matrix(d, d))
    elif args.Length = 2 then
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some arg0, Some arg1 ->
            checkNegativeDimArg arg0 env warnings ctx line getIntervalFn
            checkNegativeDimArg arg1 env warnings ctx line getIntervalFn
            let r = exprToDimIrCtx arg0 env (Some ctx)
            let c = exprToDimIrCtx arg1 env (Some ctx)
            Some (Matrix(r, c))
        | _ -> None
    else None


let private handleMatrixConstructor
    (fname: string)
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape option =
    if args.Length = 0 then Some Scalar
    elif args.Length = 1 then
        match unwrapArg args.[0] with
        | None -> None
        | Some arg ->
            checkNegativeDimArg arg env warnings ctx line getIntervalFn
            let d = exprToDimIrCtx arg env (Some ctx)
            Some (Matrix(d, d))
    elif args.Length = 2 then
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some arg0, Some arg1 ->
            checkNegativeDimArg arg0 env warnings ctx line getIntervalFn
            checkNegativeDimArg arg1 env warnings ctx line getIntervalFn
            let r = exprToDimIrCtx arg0 env (Some ctx)
            let c = exprToDimIrCtx arg1 env (Some ctx)
            Some (Matrix(r, c))
        | _ -> None
    else None


let private handleSize
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        match unwrapArg args.[0] with
        | Some arg ->
            evalExprFn arg env warnings ctx |> ignore
            Some (Matrix(Concrete 1, Concrete 2))
        | None -> None
    elif args.Length = 2 then
        match unwrapArg args.[0] with
        | Some arg ->
            evalExprFn arg env warnings ctx |> ignore
            Some Scalar
        | None -> None
    else None


let private handleCellConstructor
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape option =
    if args.Length = 1 then
        match unwrapArg args.[0] with
        | None -> None
        | Some arg ->
            checkNegativeDimArg arg env warnings ctx line getIntervalFn
            let d = exprToDimIrCtx arg env (Some ctx)
            Some (Cell(d, d, None))
    elif args.Length = 2 then
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some arg0, Some arg1 ->
            checkNegativeDimArg arg0 env warnings ctx line getIntervalFn
            checkNegativeDimArg arg1 env warnings ctx line getIntervalFn
            let r = exprToDimIrCtx arg0 env (Some ctx)
            let c = exprToDimIrCtx arg1 env (Some ctx)
            Some (Cell(r, c, None))
        | _ -> None
    else None


let private handlePassthrough
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        Some (evalArgShape args.[0] env warnings ctx evalExprFn)
    else None


let private handleTransposeFn
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        match argShape with
        | Matrix(r, c) -> Some (Matrix(c, r))
        | _ -> Some argShape
    else None


let private handleScalarQuery
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length >= 1 then
        evalArgShape args.[0] env warnings ctx evalExprFn |> ignore
        Some Scalar
    else None


let private handleReshape
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 3 then
        match unwrapArg args.[0], unwrapArg args.[1], unwrapArg args.[2] with
        | Some a0, Some a1, Some a2 ->
            let inputShape = evalExprFn a0 env warnings ctx
            let m = exprToDimIrCtx a1 env (Some ctx)
            let n = exprToDimIrCtx a2 env (Some ctx)
            // Conformability check
            if not (isUnknown inputShape) then
                let inputCount =
                    if isScalar inputShape then Some (Concrete 1)
                    elif isMatrix inputShape then
                        match inputShape with
                        | Matrix(r, c) -> Some (mulDim r c)
                        | _ -> None
                    else None
                let outputCount = mulDim m n
                match inputCount with
                | Some ic when outputCount <> Unknown ->
                    if dimsDefinitelyConflict ic outputCount then
                        let mStr = dimStr m
                        let nStr = dimStr n
                        warnings.Add(warnReshapeMismatch line inputShape mStr nStr)
                | _ -> ()
            if m = Unknown || n = Unknown then None
            else Some (Matrix(m, n))
        | _ -> None
    else None


let private handleRepmat
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 3 then
        match unwrapArg args.[0], unwrapArg args.[1], unwrapArg args.[2] with
        | Some a0, Some a1, Some a2 ->
            let aShape = evalExprFn a0 env warnings ctx
            let m = exprToDimIrCtx a1 env (Some ctx)
            let n = exprToDimIrCtx a2 env (Some ctx)
            if isUnknown aShape then Some UnknownShape
            else
                let aRows, aCols =
                    if isScalar aShape then Concrete 1, Concrete 1
                    elif isMatrix aShape then match aShape with Matrix(r, c) -> r, c | _ -> Unknown, Unknown
                    else Unknown, Unknown
                Some (Matrix(mulDim aRows m, mulDim aCols n))
        | _ -> None
    else None


let private handleDiag
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        match argShape with
        | Scalar -> Some (Matrix(Concrete 1, Concrete 1))
        | Matrix(r, c) ->
            if r = Concrete 1 then Some (Matrix(c, c))
            elif c = Concrete 1 then Some (Matrix(r, r))
            else Some (Matrix(Unknown, Concrete 1))
        | _ -> Some UnknownShape
    else None


let private handleInv
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        match argShape with
        | Matrix(r, c) when r = c -> Some (Matrix(r, c))
        | Matrix _ -> Some UnknownShape
        | _ -> Some UnknownShape
    else None


let private handleLinspace
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 2 then
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            evalExprFn a0 env warnings ctx |> ignore
            evalExprFn a1 env warnings ctx |> ignore
            Some (Matrix(Concrete 1, Concrete 100))
        | _ -> None
    elif args.Length = 3 then
        match unwrapArg args.[0], unwrapArg args.[1], unwrapArg args.[2] with
        | Some a0, Some a1, Some a2 ->
            evalExprFn a0 env warnings ctx |> ignore
            evalExprFn a1 env warnings ctx |> ignore
            let n = exprToDimIrCtx a2 env (Some ctx)
            Some (Matrix(Concrete 1, n))
        | _ -> None
    else None


let private handleReduction
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        if isScalar argShape then Some Scalar
        elif isMatrix argShape then
            match argShape with
            | Matrix(_, c) -> Some (Matrix(Concrete 1, c))
            | _ -> Some UnknownShape
        else Some UnknownShape
    elif args.Length = 2 then
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        match unwrapArg args.[1] with
        | Some dimExpr ->
            let dimVal = exprToDimIrCtx dimExpr env (Some ctx)
            match dimVal with
            | Concrete 1 ->
                if isMatrix argShape then
                    match argShape with
                    | Matrix(_, c) -> Some (Matrix(Concrete 1, c))
                    | _ -> Some UnknownShape
                else Some UnknownShape
            | Concrete 2 ->
                if isMatrix argShape then
                    match argShape with
                    | Matrix(r, _) -> Some (Matrix(r, Concrete 1))
                    | _ -> Some UnknownShape
                else Some UnknownShape
            | _ -> None
        | None -> None
    else None


let private handleMinmax
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        handleReduction args env warnings ctx evalExprFn
    elif args.Length = 2 then
        // Elementwise binary: scalar broadcasting, join dims
        let s1 = evalArgShape args.[0] env warnings ctx evalExprFn
        let s2 = evalArgShape args.[1] env warnings ctx evalExprFn
        if isScalar s1 then Some s2
        elif isScalar s2 then Some s1
        elif isMatrix s1 && isMatrix s2 then
            match s1, s2 with
            | Matrix(r1, c1), Matrix(r2, c2) ->
                if dimsDefinitelyConflict r1 r2 || dimsDefinitelyConflict c1 c2 then Some UnknownShape
                else Some (Matrix(joinDim r1 r2, joinDim c1 c2))
            | _ -> Some UnknownShape
        else Some UnknownShape
    else None


let private handleElementwise2arg
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length <> 2 then None
    else
        let s1 = evalArgShape args.[0] env warnings ctx evalExprFn
        let s2 = evalArgShape args.[1] env warnings ctx evalExprFn
        if isScalar s1 then Some s2
        elif isScalar s2 then Some s1
        elif isMatrix s1 && isMatrix s2 then
            match s1, s2 with
            | Matrix(r1, c1), Matrix(r2, c2) ->
                if dimsDefinitelyConflict r1 r2 || dimsDefinitelyConflict c1 c2 then Some UnknownShape
                else Some (Matrix(joinDim r1 r2, joinDim c1 c2))
            | _ -> Some UnknownShape
        else Some UnknownShape


let private handleDiff
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length <> 1 then None
    else
        let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
        if isScalar argShape then Some Scalar
        elif isMatrix argShape then
            match argShape with
            | Matrix(r, c) ->
                if r = Concrete 1 then Some (Matrix(Concrete 1, subDim c (Concrete 1)))
                else Some (Matrix(subDim r (Concrete 1), c))
            | _ -> Some UnknownShape
        else Some UnknownShape


let private handleKron
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length <> 2 then None
    else
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            let s1 = evalExprFn a0 env warnings ctx
            let s2 = evalExprFn a1 env warnings ctx
            if isUnknown s1 || isUnknown s2 then Some UnknownShape
            else
                let r1, c1 =
                    if isScalar s1 then Concrete 1, Concrete 1
                    elif isMatrix s1 then match s1 with Matrix(r, c) -> r, c | _ -> Unknown, Unknown
                    else Unknown, Unknown
                let r2, c2 =
                    if isScalar s2 then Concrete 1, Concrete 1
                    elif isMatrix s2 then match s2 with Matrix(r, c) -> r, c | _ -> Unknown, Unknown
                    else Unknown, Unknown
                if r1 = Unknown || r2 = Unknown then Some UnknownShape
                else Some (Matrix(mulDim r1 r2, mulDim c1 c2))
        | _ -> None


let private handleBlkdiag
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.IsEmpty then None
    else
        let mutable totalRows : Dim = Concrete 0
        let mutable totalCols : Dim = Concrete 0
        let mutable failed = false
        for arg in args do
            if not failed then
                match unwrapArg arg with
                | None -> failed <- true
                | Some a ->
                    let s = evalExprFn a env warnings ctx
                    if isUnknown s then failed <- true
                    else
                        let r, c =
                            if isScalar s then Concrete 1, Concrete 1
                            elif isMatrix s then match s with Matrix(r, c) -> r, c | _ -> Unknown, Unknown
                            else Unknown, Unknown
                        totalRows <- addDim totalRows r
                        totalCols <- addDim totalCols c
        if failed then Some UnknownShape
        else Some (Matrix(totalRows, totalCols))


let private handleStringReturn
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    for arg in args do
        evalArgShape arg env warnings ctx evalExprFn |> ignore
    Some StringShape


let private handleFind
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length = 1 then
        evalArgShape args.[0] env warnings ctx evalExprFn |> ignore
        Some (Matrix(Concrete 1, Unknown))
    else None


let private handleEigSingle
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length <> 1 then None
    else
        match unwrapArg args.[0] with
        | None -> None
        | Some a ->
            let s = evalExprFn a env warnings ctx
            if isScalar s then Some Scalar
            elif isMatrix s then
                match s with
                | Matrix(r, c) when r = c -> Some (Matrix(r, Concrete 1))
                | Matrix _ -> Some (Matrix(Unknown, Concrete 1))
                | _ -> Some UnknownShape
            else Some UnknownShape


let private handleSvdSingle
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length <> 1 then None
    else
        match unwrapArg args.[0] with
        | None -> None
        | Some a ->
            evalExprFn a env warnings ctx |> ignore
            Some (Matrix(Unknown, Concrete 1))


let private handleCat
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length < 2 then None
    else
        match unwrapArg args.[0] with
        | None -> None
        | Some dimExpr ->
            let dimVal = exprToDimIrCtx dimExpr env (Some ctx)
            match dimVal with
            | Concrete 1 | Concrete 2 ->
                let restArgs = args |> List.tail
                let shapes =
                    restArgs |> List.choose (fun arg ->
                        match unwrapArg arg with
                        | None -> None
                        | Some a ->
                            let s = evalExprFn a env warnings ctx
                            if isUnknown s then None else Some s)
                if shapes.Length <> restArgs.Length then Some UnknownShape
                elif shapes.IsEmpty then None
                else
                    let normalize s =
                        if isScalar s then Some (Concrete 1, Concrete 1)
                        elif isMatrix s then match s with Matrix(r, c) -> Some (r, c) | _ -> None
                        else None
                    let normalized = shapes |> List.choose normalize
                    if normalized.Length <> shapes.Length then Some UnknownShape
                    else
                        let (r0, c0) = normalized.[0]
                        let (rFinal, cFinal) =
                            normalized.[1..] |> List.fold (fun (rAcc, cAcc) (r, c) ->
                                match dimVal with
                                | Concrete 1 -> (addDim rAcc r, joinDim cAcc c)
                                | _ ->          (joinDim rAcc r, addDim cAcc c)) (r0, c0)
                        Some (Matrix(rFinal, cFinal))
            | _ -> None


let private handleRandi
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape option =
    if args.Length < 1 then None
    else
        // Evaluate first arg (imax) for side effects
        match unwrapArg args.[0] with
        | None -> None
        | Some imax ->
            evalExprFn imax env warnings ctx |> ignore
            if args.Length = 1 then Some Scalar
            elif args.Length = 2 then
                match unwrapArg args.[1] with
                | None -> None
                | Some arg ->
                    checkNegativeDimArg arg env warnings ctx line getIntervalFn
                    let d = exprToDimIrCtx arg env (Some ctx)
                    Some (Matrix(d, d))
            elif args.Length = 3 then
                match unwrapArg args.[1], unwrapArg args.[2] with
                | Some arg0, Some arg1 ->
                    checkNegativeDimArg arg0 env warnings ctx line getIntervalFn
                    checkNegativeDimArg arg1 env warnings ctx line getIntervalFn
                    let r = exprToDimIrCtx arg0 env (Some ctx)
                    let c = exprToDimIrCtx arg1 env (Some ctx)
                    Some (Matrix(r, c))
                | _ -> None
            else None


let private handleFft
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length >= 1 then
        match unwrapArg args.[0] with
        | Some a ->
            let s = evalExprFn a env warnings ctx
            if isScalar s || isMatrix s then Some s else None
        | None -> None
    else None


let private handleSparseFull
    (fname: string)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    // sparse(m, n) — constructor form
    if fname = "sparse" && args.Length = 2 then
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            let r = exprToDimIrCtx a0 env (Some ctx)
            let c = exprToDimIrCtx a1 env (Some ctx)
            Some (Matrix(r, c))
        | _ -> None
    elif args.Length = 1 then
        match unwrapArg args.[0] with
        | Some a ->
            let s = evalExprFn a env warnings ctx
            if isScalar s || isMatrix s then Some s else None
        | None -> None
    else None


let private handleCross
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length >= 1 then
        match unwrapArg args.[0] with
        | Some a -> Some (evalExprFn a env warnings ctx)
        | None -> None
    else None


let private handleConv
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length < 2 then Some (Matrix(Unknown, Concrete 1))
    else
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            let s0 = evalExprFn a0 env warnings ctx
            let s1 = evalExprFn a1 env warnings ctx
            // Determine length and orientation from each input
            let isRowVec0, len0 =
                match s0 with
                | Matrix(r, Concrete 1) -> false, Some r
                | Matrix(Concrete 1, c) -> true, Some c
                | _ -> false, (if isScalar s0 then Some (Concrete 1) else None)
            let isRowVec1, len1 =
                match s1 with
                | Matrix(r, Concrete 1) -> false, Some r
                | Matrix(Concrete 1, c) -> true, Some c
                | _ -> false, (if isScalar s1 then Some (Concrete 1) else None)
            match len0, len1 with
            | Some m, Some n ->
                let outLen = addDim (addDim m n) (Concrete -1)
                // Return row vector only if both inputs are row vectors
                if isRowVec0 && isRowVec1 then Some (Matrix(Concrete 1, outLen))
                else Some (Matrix(outLen, Concrete 1))
            | _ -> Some (Matrix(Unknown, Concrete 1))
        | _ -> Some (Matrix(Unknown, Concrete 1))


let private handleFilter
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length < 3 then Some UnknownShape
    else
        match unwrapArg args.[0], unwrapArg args.[1], unwrapArg args.[2] with
        | Some a0, Some a1, Some a2 ->
            evalExprFn a0 env warnings ctx |> ignore
            evalExprFn a1 env warnings ctx |> ignore
            Some (evalExprFn a2 env warnings ctx)
        | _ -> Some UnknownShape


let private handleWindowFunction
    (args: IndexArg list)
    (env: Env)
    (ctx: AnalysisContext)
    : Shape option =
    if args.Length >= 1 then
        match unwrapArg args.[0] with
        | Some a -> Some (Matrix(exprToDimIrCtx a env (Some ctx), Concrete 1))
        | None -> Some (Matrix(Unknown, Concrete 1))
    else Some (Matrix(Unknown, Concrete 1))


let private handleFilterDesign
    (args: IndexArg list)
    (_env: Env)
    (_warnings: ResizeArray<Diagnostic>)
    (_ctx: AnalysisContext)
    (_evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    ignore args
    Some (Matrix(Concrete 1, Unknown))


let private handleXcorr
    (_args: IndexArg list)
    (_env: Env)
    (_warnings: ResizeArray<Diagnostic>)
    (_ctx: AnalysisContext)
    (_evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    Some (Matrix(Unknown, Concrete 1))


let private handleDcm
    (_args: IndexArg list)
    (_env: Env)
    (_warnings: ResizeArray<Diagnostic>)
    (_ctx: AnalysisContext)
    (_evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    Some (Matrix(Concrete 3, Concrete 3))


let private handleQuat
    (_args: IndexArg list)
    (_env: Env)
    (_warnings: ResizeArray<Diagnostic>)
    (_ctx: AnalysisContext)
    (_evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    Some (Matrix(Concrete 1, Concrete 4))


let private handlePolyfit
    (args: IndexArg list)
    (env: Env)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    ignore evalExprFn
    if args.Length >= 3 then
        match unwrapArg args.[2] with
        | Some a2 ->
            let nDim = exprToDimIrCtx a2 env (Some ctx)
            Some (Matrix(Concrete 1, addDim nDim (Concrete 1)))
        | None -> Some (Matrix(Concrete 1, Unknown))
    else Some (Matrix(Concrete 1, Unknown))


let private handlePolyval
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length >= 2 then
        match unwrapArg args.[1] with
        | Some a -> Some (evalExprFn a env warnings ctx)
        | None -> None
    else None


let private handleMeshgrid (_args: IndexArg list) (_env: Env) (_w: _) (_ctx: _) (_eval: _) : Shape option =
    Some (Matrix(Unknown, Unknown))


let private handleStruct
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.IsEmpty then Some (Struct([], false))
    else
        let mutable fields : (string * Shape) list = []
        let mutable i = 0
        let mutable ok = true
        while ok && i + 1 < args.Length do
            match unwrapArg args.[i], unwrapArg args.[i + 1] with
            | Some keyExpr, Some valExpr ->
                match keyExpr with
                | StringLit(_, _, fieldName) ->
                    let valShape = evalExprFn valExpr env warnings ctx
                    fields <- fields @ [ (fieldName, valShape) ]
                | _ -> ok <- false
            | _ -> ok <- false
            i <- i + 2
        if ok then Some (Struct(fields, false))
        else Some (Struct([], false))


let private handleFieldnames (_args: IndexArg list) : Shape option =
    Some (Cell(Unknown, Concrete 1, None))


let private handleNdims (_args: IndexArg list) : Shape option =
    Some Scalar


let private handleSub2ind
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length >= 2 then
        match unwrapArg args.[1] with
        | Some a -> Some (evalExprFn a env warnings ctx)
        | None -> Some Scalar
    else Some Scalar


let private handleHorzcatVertcat
    (fname: string)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.IsEmpty then Some (Matrix(Concrete 0, Concrete 0))
    else
        let shapes =
            args |> List.choose (fun arg ->
                match unwrapArg arg with
                | Some a -> Some (evalExprFn a env warnings ctx)
                | None -> None)
        if shapes.IsEmpty then Some UnknownShape
        elif fname = "horzcat" then
            let r =
                match shapes.[0] with
                | Matrix(r, _) -> r
                | _ -> if isScalar shapes.[0] then Concrete 1 else Unknown
            let mutable c : Dim = Concrete 0
            for s in shapes do
                let sc = match s with Matrix(_, c) -> c | _ -> if isScalar s then Concrete 1 else Unknown
                c <- addDim c sc
            Some (Matrix(r, c))
        else // vertcat
            let col =
                match shapes.[0] with
                | Matrix(_, c) -> c
                | _ -> if isScalar shapes.[0] then Concrete 1 else Unknown
            let mutable r : Dim = Concrete 0
            for s in shapes do
                let sr = match s with Matrix(r, _) -> r | _ -> if isScalar s then Concrete 1 else Unknown
                r <- addDim r sr
            Some (Matrix(r, col))


// ---------------------------------------------------------------------------
// Multi-return handler helpers
// ---------------------------------------------------------------------------

let private evalFirstArgShape
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : (Dim * Dim) option =
    if args.IsEmpty then None
    else
        match unwrapArg args.[0] with
        | None -> None
        | Some a ->
            let s = evalExprFn a env warnings ctx
            if isScalar s then Some (Concrete 1, Concrete 1)
            elif isMatrix s then match s with Matrix(r, c) -> Some (r, c) | _ -> None
            else None


// ---------------------------------------------------------------------------
// Control System Toolbox helpers (defined here to use evalFirstArgShape above)
// ---------------------------------------------------------------------------

/// Extract (n, m) from A(n×n) and B(n×m) -- shared by lqr/dlqr/place/acker/care/dare/ctrb
let private extractStateFeedbackDims
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : (Dim * Dim) option =
    if args.Length < 2 then None
    else
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            let sA = evalExprFn a0 env warnings ctx
            let sB = evalExprFn a1 env warnings ctx
            let n = match sA with Matrix(r, _) -> r | _ -> if isScalar sA then Concrete 1 else Unknown
            let m = match sB with Matrix(_, c) -> c | _ -> if isScalar sB then Concrete 1 else Unknown
            Some (n, m)
        | _ -> None


/// lqr/dlqr/place/acker single-return: K = matrix[m x n]
let private handleGainMatrix
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    match extractStateFeedbackDims args env warnings ctx evalExprFn with
    | Some (n, m) -> Some (Matrix(m, n))
    | None -> Some UnknownShape


/// lyap/dlyap single-return: X = matrix[n x n] (passthrough square dim from first arg)
let private handleSquarePassthrough
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    match evalFirstArgShape args env warnings ctx evalExprFn with
    | Some (r, c) ->
        let n = if r = c then r else joinDim r c
        Some (Matrix(n, n))
    | None -> Some UnknownShape


/// care/dare single-return: X = matrix[n x n]
let private handleCareSquare
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    match extractStateFeedbackDims args env warnings ctx evalExprFn with
    | Some (n, _) -> Some (Matrix(n, n))
    | None -> Some UnknownShape


/// obsv(A, C): A n×n, C p×n -- returns matrix[n*p x n]
let private handleObsv
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    if args.Length < 2 then Some UnknownShape
    else
        match unwrapArg args.[0], unwrapArg args.[1] with
        | Some a0, Some a1 ->
            let sA = evalExprFn a0 env warnings ctx
            let sC = evalExprFn a1 env warnings ctx
            let n = match sA with Matrix(r, _) -> r | _ -> if isScalar sA then Concrete 1 else Unknown
            let p = match sC with Matrix(r, _) -> r | _ -> if isScalar sC then Concrete 1 else Unknown
            Some (Matrix(mulDim n p, n))
        | _ -> Some UnknownShape


/// ctrb(A, B): A n×n, B n×m -- returns matrix[n x n*m]
let private handleCtrb
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape option =
    match extractStateFeedbackDims args env warnings ctx evalExprFn with
    | Some (n, m) -> Some (Matrix(n, mulDim n m))
    | None -> Some UnknownShape


// Multi-return handlers
let private handleMultiEig
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        match evalFirstArgShape args env warnings ctx evalExprFn with
        | None -> Some [ UnknownShape; UnknownShape ]
        | Some (r, c) ->
            let n = if r = c then r else joinDim r c
            Some [ Matrix(n, n); Matrix(n, n) ]


let private handleMultiSvd
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 3 then None
    else
        match evalFirstArgShape args env warnings ctx evalExprFn with
        | None -> Some [ UnknownShape; UnknownShape; UnknownShape ]
        | Some (m, n) -> Some [ Matrix(m, m); Matrix(m, n); Matrix(n, n) ]


let private handleMultiLu
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    match evalFirstArgShape args env warnings ctx evalExprFn with
    | None ->
        if numTargets = 2 then Some [ UnknownShape; UnknownShape ]
        elif numTargets = 3 then Some [ UnknownShape; UnknownShape; UnknownShape ]
        else None
    | Some (m, n) ->
        if numTargets = 2 then Some [ Matrix(m, m); Matrix(m, n) ]
        elif numTargets = 3 then Some [ Matrix(m, n); Matrix(n, n); Matrix(m, m) ]
        else None


let private handleMultiQr
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        match evalFirstArgShape args env warnings ctx evalExprFn with
        | None -> Some [ UnknownShape; UnknownShape ]
        | Some (m, n) -> Some [ Matrix(m, m); Matrix(m, n) ]


let private handleMultiChol
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        match evalFirstArgShape args env warnings ctx evalExprFn with
        | None -> Some [ UnknownShape; Scalar ]
        | Some (r, c) ->
            let n = if r = c then r else joinDim r c
            Some [ Matrix(n, n); Scalar ]


let private handleMultiSize
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        evalFirstArgShape args env warnings ctx evalExprFn |> ignore
        Some [ Scalar; Scalar ]


let private handleMultiSort
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        if args.IsEmpty then Some [ UnknownShape; UnknownShape ]
        else
            match unwrapArg args.[0] with
            | None -> Some [ UnknownShape; UnknownShape ]
            | Some a ->
                let s = evalExprFn a env warnings ctx
                Some [ s; s ]


let private handleMultiFind
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    evalFirstArgShape args env warnings ctx evalExprFn |> ignore
    if numTargets = 2 then Some [ Matrix(Concrete 1, Unknown); Matrix(Concrete 1, Unknown) ]
    elif numTargets = 3 then Some [ Matrix(Concrete 1, Unknown); Matrix(Concrete 1, Unknown); Matrix(Concrete 1, Unknown) ]
    else None


let private handleMultiUnique
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    evalFirstArgShape args env warnings ctx evalExprFn |> ignore
    if numTargets = 2 then Some [ Matrix(Concrete 1, Unknown); Matrix(Unknown, Concrete 1) ]
    elif numTargets = 3 then Some [ Matrix(Concrete 1, Unknown); Matrix(Unknown, Concrete 1); Matrix(Unknown, Concrete 1) ]
    else None


let private handleMultiMinmax
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        if args.IsEmpty then Some [ UnknownShape; UnknownShape ]
        else
            let argShape = evalArgShape args.[0] env warnings ctx evalExprFn
            if isScalar argShape then Some [ Scalar; Scalar ]
            elif isMatrix argShape then
                match argShape with
                | Matrix(_, c) ->
                    let reduction = Matrix(Concrete 1, c)
                    Some [ reduction; reduction ]
                | _ -> Some [ UnknownShape; UnknownShape ]
            else Some [ UnknownShape; UnknownShape ]


let private handleMultiFileparts (_args: IndexArg list) (numTargets: int) : Shape list option =
    if numTargets <= 3 then Some (List.replicate numTargets StringShape)
    else None


let private handleMultiFopen (_args: IndexArg list) (numTargets: int) : Shape list option =
    if numTargets = 2 then Some [ Scalar; StringShape ]
    else None


let private handleMultiMeshgrid (_args: IndexArg list) (numTargets: int) : Shape list option =
    if numTargets = 2 || numTargets = 3 then Some (List.replicate numTargets UnknownShape)
    else None


let private handleMultiAny (numTargets: int) : Shape list option =
    Some (List.replicate numTargets UnknownShape)


/// lqr/dlqr multi-return [K, S, e]: K=matrix[m×n], S=matrix[n×n], e=matrix[n×1]
let private handleMultiLqr
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 3 then None
    else
        match extractStateFeedbackDims args env warnings ctx evalExprFn with
        | Some (n, m) -> Some [ Matrix(m, n); Matrix(n, n); Matrix(n, Concrete 1) ]
        | None -> Some [ UnknownShape; UnknownShape; UnknownShape ]


/// care/dare multi-return [X, L, G]: X=matrix[n×n], L=matrix[n×1], G=matrix[m×n]
let private handleMultiCare
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 3 then None
    else
        match extractStateFeedbackDims args env warnings ctx evalExprFn with
        | Some (n, m) -> Some [ Matrix(n, n); Matrix(n, Concrete 1); Matrix(m, n) ]
        | None -> Some [ UnknownShape; UnknownShape; UnknownShape ]


/// butter/cheby1/cheby2/ellip/besself multi-return [b, a]: both matrix[1 x (n+1)]
let private handleMultiFilterDesign
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 2 then None
    else
        let d =
            if args.IsEmpty then Unknown
            else
                match unwrapArg args.[0] with
                | Some a -> evalExprFn a env warnings ctx |> ignore; exprToDimIrCtx a env (Some ctx)
                | None -> Unknown
        let outLen = addDim d (Concrete 1)
        Some [ Matrix(Concrete 1, outLen); Matrix(Concrete 1, outLen) ]


/// dcm2angle multi-return [r1, r2, r3]: all Scalar
let private handleMultiDcm2angle
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (numTargets: int)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    : Shape list option =
    if numTargets <> 3 then None
    else
        if not args.IsEmpty then
            match unwrapArg args.[0] with
            | Some a -> evalExprFn a env warnings ctx |> ignore
            | None -> ()
        Some [ Scalar; Scalar; Scalar ]


// Supported forms lookup for count mismatch messages
let MULTI_SUPPORTED_FORMS : Map<string, string> =
    Map.ofList [
        "eig", "1 or 2"; "svd", "1 or 3"; "lu", "2 or 3"; "qr", "2"
        "chol", "2"; "size", "2"; "sort", "2"; "find", "1, 2, or 3"
        "unique", "1, 2, or 3"; "min", "1 or 2"; "max", "1 or 2"
        "fileparts", "1-3"; "fopen", "2"; "meshgrid", "2 or 3"
        "cellfun", "any"; "ndgrid", "any"; "regexp", "any"; "regexpi", "any"
        "lqr", "1 or 3"; "dlqr", "1 or 3"; "care", "1 or 3"; "dare", "1 or 3"
        "butter", "1 or 2"; "cheby1", "1 or 2"; "cheby2", "1 or 2"; "ellip", "1 or 2"; "besself", "1 or 2"
        "dcm2angle", "1 or 3"
    ]


// ---------------------------------------------------------------------------
// Main dispatch functions
// ---------------------------------------------------------------------------

/// evalBuiltinCall: dispatch a builtin call, returning inferred shape.
/// evalExprFn and getIntervalFn break the circular dependency with EvalExpr.
let evalBuiltinCall
    (fname: string)
    (line: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape =

    // Complex handlers take priority (checked first)
    let complexResult : Shape option =
        match fname with
        | "zeros" | "ones" -> handleZerosOnes fname line args env warnings ctx evalExprFn getIntervalFn
        | "size"           -> handleSize line args env warnings ctx evalExprFn
        | "cell"           -> handleCellConstructor line args env warnings ctx getIntervalFn
        | "transpose"      -> handleTransposeFn args env warnings ctx evalExprFn
        | "reshape"        -> handleReshape line args env warnings ctx evalExprFn
        | "repmat"         -> handleRepmat args env warnings ctx evalExprFn
        | "diag"           -> handleDiag args env warnings ctx evalExprFn
        | "inv" | "pinv"   -> handleInv args env warnings ctx evalExprFn
        | "linspace" | "logspace" -> handleLinspace args env warnings ctx evalExprFn
        | "min" | "max"    -> handleMinmax args env warnings ctx evalExprFn
        | "mod" | "rem" | "atan2" | "power" | "hypot" | "xor" -> handleElementwise2arg args env warnings ctx evalExprFn
        | "diff"           -> handleDiff args env warnings ctx evalExprFn
        | "kron"           -> handleKron args env warnings ctx evalExprFn
        | "blkdiag"        -> handleBlkdiag args env warnings ctx evalExprFn
        | "randi"          -> handleRandi line args env warnings ctx evalExprFn getIntervalFn
        | "find"           -> handleFind args env warnings ctx evalExprFn
        | "cat"            -> handleCat args env warnings ctx evalExprFn
        | "eig"            -> handleEigSingle args env warnings ctx evalExprFn
        | "svd"            -> handleSvdSingle args env warnings ctx evalExprFn
        | "fft" | "ifft" | "fft2" | "ifft2" -> handleFft args env warnings ctx evalExprFn
        | "sparse" | "full" -> handleSparseFull fname args env warnings ctx evalExprFn
        | "cross"          -> handleCross args env warnings ctx evalExprFn
        | "conv"           -> handleConv args env warnings ctx evalExprFn
        | "deconv"         -> Some (Matrix(Unknown, Concrete 1))
        | "polyfit"        -> handlePolyfit args env ctx evalExprFn
        | "polyval" | "interp1" -> handlePolyval args env warnings ctx evalExprFn
        | "meshgrid"       -> handleMeshgrid args env warnings ctx evalExprFn
        | "struct"         -> handleStruct args env warnings ctx evalExprFn
        | "fieldnames"     -> handleFieldnames args
        | "ndims"          -> handleNdims args
        | "sub2ind"        -> handleSub2ind args env warnings ctx evalExprFn
        | "horzcat" | "vertcat" -> handleHorzcatVertcat fname args env warnings ctx evalExprFn
        | "lqr" | "dlqr" | "place" | "acker" -> handleGainMatrix args env warnings ctx evalExprFn
        | "care" | "dare"  -> handleCareSquare args env warnings ctx evalExprFn
        | "lyap" | "dlyap" -> handleSquarePassthrough args env warnings ctx evalExprFn
        | "obsv"           -> handleObsv args env warnings ctx evalExprFn
        | "ctrb"           -> handleCtrb args env warnings ctx evalExprFn
        | "filter" | "filtfilt" -> handleFilter args env warnings ctx evalExprFn
        | "hamming" | "hann" | "blackman" | "kaiser" | "rectwin" | "bartlett" ->
            handleWindowFunction args env ctx
        | "butter" | "cheby1" | "cheby2" | "ellip" | "besself" ->
            handleFilterDesign args env warnings ctx evalExprFn
        | "xcorr"          -> handleXcorr args env warnings ctx evalExprFn
        | "angle2dcm" | "quat2dcm" | "eul2rotm" | "quat2rotm" ->
            handleDcm args env warnings ctx evalExprFn
        | "dcm2quat" | "rotm2quat" | "quatmultiply" ->
            handleQuat args env warnings ctx evalExprFn
        | "dcm2angle"      -> Some Scalar
        | _                -> None

    match complexResult with
    | Some s -> s
    | None ->
        // Declarative dispatch
        if Set.contains fname PASSTHROUGH_BUILTINS || Set.contains fname TYPE_CAST_BUILTINS then
            if args.Length = 1 then evalArgShape args.[0] env warnings ctx evalExprFn
            else UnknownShape
        elif Set.contains fname SCALAR_PREDICATE_BUILTINS then
            if args.Length >= 1 then
                match unwrapArg args.[0] with
                | Some a -> evalExprFn a env warnings ctx |> ignore
                | None -> ()
            Scalar
        elif Set.contains fname SCALAR_QUERY_BUILTINS then
            if args.Length >= 1 then evalArgShape args.[0] env warnings ctx evalExprFn |> ignore
            Scalar
        elif Set.contains fname SCALAR_NARY_BUILTINS then
            Scalar
        elif Set.contains fname STRING_RETURN_BUILTINS then
            for arg in args do evalArgShape arg env warnings ctx evalExprFn |> ignore
            StringShape
        elif Set.contains fname REDUCTION_BUILTINS then
            handleReduction args env warnings ctx evalExprFn |> Option.defaultValue UnknownShape
        elif Set.contains fname MATRIX_CONSTRUCTOR_BUILTINS then
            handleMatrixConstructor fname line args env warnings ctx evalExprFn getIntervalFn |> Option.defaultValue UnknownShape
        else
            // Known builtin without a matching shape rule
            UnknownShape


/// evalMultiBuiltinCall: dispatch a multi-return builtin call, returning shape list.
let evalMultiBuiltinCall
    (fname: string)
    (numTargets: int)
    (args: IndexArg list)
    (env: Env)
    (warnings: ResizeArray<Diagnostic>)
    (ctx: AnalysisContext)
    (evalExprFn: Expr -> Env -> ResizeArray<Diagnostic> -> AnalysisContext -> Shape)
    (getIntervalFn: Expr -> Env -> AnalysisContext -> Interval option)
    : Shape list option =

    match fname with
    | "eig"      -> handleMultiEig     args env warnings ctx numTargets evalExprFn
    | "svd"      -> handleMultiSvd     args env warnings ctx numTargets evalExprFn
    | "lu"       -> handleMultiLu      args env warnings ctx numTargets evalExprFn
    | "qr"       -> handleMultiQr      args env warnings ctx numTargets evalExprFn
    | "chol"     -> handleMultiChol    args env warnings ctx numTargets evalExprFn
    | "size"     -> handleMultiSize    args env warnings ctx numTargets evalExprFn
    | "sort"     -> handleMultiSort    args env warnings ctx numTargets evalExprFn
    | "find"     -> handleMultiFind    args env warnings ctx numTargets evalExprFn
    | "unique"   -> handleMultiUnique  args env warnings ctx numTargets evalExprFn
    | "min" | "max" -> handleMultiMinmax args env warnings ctx numTargets evalExprFn
    | "fileparts" -> handleMultiFileparts args numTargets
    | "fopen"    -> handleMultiFopen   args numTargets
    | "meshgrid" -> handleMultiMeshgrid args numTargets
    | "cellfun" | "ndgrid" | "regexp" | "regexpi" -> handleMultiAny numTargets
    | "lqr" | "dlqr"   -> handleMultiLqr  args env warnings ctx numTargets evalExprFn
    | "care" | "dare"   -> handleMultiCare args env warnings ctx numTargets evalExprFn
    | "butter" | "cheby1" | "cheby2" | "ellip" | "besself" ->
        handleMultiFilterDesign args env warnings ctx numTargets evalExprFn
    | "dcm2angle" -> handleMultiDcm2angle args env warnings ctx numTargets evalExprFn
    | _          -> None
