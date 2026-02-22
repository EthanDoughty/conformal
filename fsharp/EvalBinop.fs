module EvalBinop

open Ir
open Shapes
open Env
open Context
open Diagnostics

// ---------------------------------------------------------------------------
// Binary operation shape inference.
// Port of analysis/eval_binop.py
// ---------------------------------------------------------------------------

/// evalBinopIr: evaluate a binary operation and infer result shape.
/// getDivisorInterval: callback to compute interval of divisor expression (for div-by-zero check).
let evalBinopIr
    (op: string)
    (left: Shape)
    (right: Shape)
    (warnings: Diagnostic list ref)
    (leftExpr: Expr)
    (rightExpr: Expr)
    (line: int)
    (ctx: AnalysisContext)
    (env: Env)
    (getDivisorInterval: Expr -> Intervals.Interval option)
    : Shape =

    // Comparison operators: always return scalar
    if Set.contains op (Set.ofList ["=="; "~="; "<"; "<="; ">"; ">="]) then
        match left, right with
        | Matrix _, Scalar | Scalar, Matrix _ ->
            warnings.Value <- warnings.Value @
                [ warnSuspiciousComparisonMatrixScalar line op leftExpr rightExpr left right ]
        | Matrix _, Matrix _ ->
            warnings.Value <- warnings.Value @
                [ warnMatrixToMatrixComparison line op leftExpr rightExpr left right ]
        | _ -> ()
        Scalar

    // Short-circuit logical operators
    elif op = "&&" || op = "||" then
        match left, right with
        | Matrix _, _ | _, Matrix _ ->
            warnings.Value <- warnings.Value @
                [ warnLogicalOpNonScalar line op leftExpr rightExpr left right ]
        | _ -> ()
        Scalar

    // Colon range: always 1 x unknown
    elif op = ":" then
        Matrix(Concrete 1, Unknown)

    // String + string = numeric row vector (MATLAB behavior)
    elif op = "+" && isString left && isString right then
        Matrix(Concrete 1, Unknown)

    // String + non-string: warning + unknown
    elif Set.contains op (Set.ofList ["+"; "-"; "*"; ".*"; "/"; "./"; ".^"; "\\"; "^"]) &&
         (isString left || isString right) then
        if not (isString left && isString right && op = "+") then
            warnings.Value <- warnings.Value @
                [ warnStringArithmetic line op left right ]
            UnknownShape
        else
            Matrix(Concrete 1, Unknown)   // fallthrough for string+string

    // Type guard: non-numeric types cannot participate in arithmetic
    elif not (isUnknown left || isUnknown right) &&
         Set.contains op (Set.ofList ["+"; "-"; "*"; ".*"; "/"; "./"; ".^"; "\\"; "^"; "&"; "|"]) &&
         (not (isNumeric left) || not (isNumeric right)) then
        warnings.Value <- warnings.Value @
            [ warnArithmeticTypeMismatch line op leftExpr rightExpr left right ]
        UnknownShape

    // Matrix power: A^n
    elif op = "^" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, _ | _, Matrix _ -> UnknownShape
        | Matrix(r, c), Scalar ->
            if dimsDefinitelyConflict r c then
                warnings.Value <- warnings.Value @
                    [ warnMatrixPowerNonSquare line leftExpr left ]
                UnknownShape
            else left
        | _ -> UnknownShape

    // mldivide: A\b
    elif op = "\\" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, _ -> right
        | _, Scalar when isMatrix left -> Scalar
        | UnknownShape, _ | _, UnknownShape -> UnknownShape
        | Matrix(lr, lc), Matrix(rr, rc) ->
            Constraints.recordConstraint ctx env lr rr line
            if dimsDefinitelyConflict lr rr then
                warnings.Value <- warnings.Value @
                    [ warnMldivideDimMismatch line leftExpr rightExpr left right ]
                UnknownShape
            else Matrix(lc, rc)
        | _ -> UnknownShape

    // Scalar broadcasting
    elif isScalar left && not (isScalar right) then right
    elif isScalar right && not (isScalar left) then left

    // Elementwise operations: +, -, .*, ./, /, .^
    elif Set.contains op (Set.ofList ["+"; "-"; ".*"; "./"; "/"; ".^"]) then
        match left, right with
        | UnknownShape, _ | _, UnknownShape -> UnknownShape
        | Scalar, Scalar ->
            // Division-by-zero check
            if op = "/" || op = "./" then
                let divisorIv = getDivisorInterval rightExpr
                if Intervals.intervalIsExactlyZero divisorIv then
                    warnings.Value <- warnings.Value @
                        [ warnDivisionByZero line leftExpr rightExpr ]
            Scalar
        | Matrix(r1, c1), Matrix(r2, c2) ->
            Constraints.recordConstraint ctx env r1 r2 line
            Constraints.recordConstraint ctx env c1 c2 line
            let rConflict = dimsDefinitelyConflict r1 r2
            let cConflict = dimsDefinitelyConflict c1 c2
            if rConflict || cConflict then
                warnings.Value <- warnings.Value @
                    [ warnElementwiseMismatch line op leftExpr rightExpr left right ]
                UnknownShape
            else Matrix(joinDim r1 r2, joinDim c1 c2)
        | _ -> UnknownShape

    // Matrix multiplication: *
    elif op = "*" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, Matrix _ -> right
        | Matrix _, Scalar -> left
        | Matrix(r1, c1), Matrix(r2, c2) ->
            Constraints.recordConstraint ctx env c1 r2 line
            if dimsDefinitelyConflict c1 r2 then
                let suggest =
                    not (dimsDefinitelyConflict r1 r2) &&
                    not (dimsDefinitelyConflict c1 c2)
                warnings.Value <- warnings.Value @
                    [ warnMatmulMismatch line leftExpr rightExpr left right suggest ]
                UnknownShape
            else Matrix(r1, c2)
        | _ -> UnknownShape

    // Element-wise logical: &, |
    elif op = "&" || op = "|" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, _ -> right
        | _, Scalar -> left
        | UnknownShape, _ | _, UnknownShape -> UnknownShape
        | Matrix(r1, c1), Matrix(r2, c2) ->
            Constraints.recordConstraint ctx env r1 r2 line
            Constraints.recordConstraint ctx env c1 c2 line
            let rConflict = dimsDefinitelyConflict r1 r2
            let cConflict = dimsDefinitelyConflict c1 c2
            if rConflict || cConflict then
                warnings.Value <- warnings.Value @
                    [ warnElementwiseMismatch line op leftExpr rightExpr left right ]
                UnknownShape
            else Matrix(joinDim r1 r2, joinDim c1 c2)
        | _ -> UnknownShape

    else UnknownShape
