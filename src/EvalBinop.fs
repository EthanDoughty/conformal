module EvalBinop

open Ir
open Shapes
open Env
open Context
open Diagnostics
open SharedTypes

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
    (warnings: ResizeArray<Diagnostic>)
    (leftExpr: Expr)
    (rightExpr: Expr)
    (line: int)
    (ctx: AnalysisContext)
    (env: Env)
    (getDivisorInterval: Expr -> Interval option)
    : Shape =

    // Comparison operators: return broadcast shape (logical array matching operand shape)
    if Set.contains op (Set.ofList ["=="; "~="; "<"; "<="; ">"; ">="]) then
        match left, right with
        | Matrix _, Scalar | Scalar, Matrix _ ->
            warnings.Add(warnSuspiciousComparisonMatrixScalar line op leftExpr rightExpr left right)
        | Matrix _, Matrix _ ->
            warnings.Add(warnMatrixToMatrixComparison line op leftExpr rightExpr left right)
        | _ -> ()
        // Return shape reflects MATLAB broadcast semantics: A > 0 where A is matrix[m x n]
        // returns a logical matrix[m x n], not a scalar.
        match left, right with
        | Scalar, Scalar -> Scalar
        | Matrix(r, c), Scalar | Scalar, Matrix(r, c) -> Matrix(r, c)
        | Matrix(r1, c1), Matrix(r2, c2) -> Matrix(joinDim r1 r2, joinDim c1 c2)
        | UnknownShape, _ | _, UnknownShape -> UnknownShape
        | _ -> Scalar  // string, struct, cell, etc: scalar boolean

    // Short-circuit logical operators
    elif op = "&&" || op = "||" then
        match left, right with
        | Matrix _, _ | _, Matrix _ ->
            warnings.Add(warnLogicalOpNonScalar line op leftExpr rightExpr left right)
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
            warnings.Add(warnStringArithmetic line op left right)
            UnknownShape
        else
            Matrix(Concrete 1, Unknown)   // fallthrough for string+string

    // Type guard: non-numeric types cannot participate in arithmetic
    elif not (isUnknown left || isUnknown right) &&
         Set.contains op (Set.ofList ["+"; "-"; "*"; ".*"; "/"; "./"; ".^"; "\\"; "^"; "&"; "|"]) &&
         (not (isNumeric left) || not (isNumeric right)) then
        warnings.Add(warnArithmeticTypeMismatch line op leftExpr rightExpr left right)
        UnknownShape

    // Matrix power: A^n
    elif op = "^" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, _ | _, Matrix _ -> UnknownShape
        | Matrix(r, c), Scalar ->
            if dimsDefinitelyConflict r c then
                warnings.Add(warnMatrixPowerNonSquare line leftExpr left)
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
            let lr' = Constraints.resolveDim ctx lr
            let rr' = Constraints.resolveDim ctx rr
            if dimsDefinitelyConflict lr' rr' then
                warnings.Add(warnMldivideDimMismatch line leftExpr rightExpr left right)
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
                    warnings.Add(warnDivisionByZero line leftExpr rightExpr)
            Scalar
        | Matrix(r1, c1), Matrix(r2, c2) ->
            Constraints.recordConstraint ctx env r1 r2 line
            Constraints.recordConstraint ctx env c1 c2 line
            let r1' = Constraints.resolveDim ctx r1
            let r2' = Constraints.resolveDim ctx r2
            let c1' = Constraints.resolveDim ctx c1
            let c2' = Constraints.resolveDim ctx c2
            let rConflict = dimsDefinitelyConflict r1' r2'
            let cConflict = dimsDefinitelyConflict c1' c2'
            if rConflict || cConflict then
                warnings.Add(warnElementwiseMismatch line op leftExpr rightExpr left right)
                UnknownShape
            else Matrix(joinDim r1' r2', joinDim c1' c2')
        | _ -> UnknownShape

    // Matrix multiplication: *
    elif op = "*" then
        match left, right with
        | Scalar, Scalar -> Scalar
        | Scalar, Matrix _ -> right
        | Matrix _, Scalar -> left
        | Matrix(r1, c1), Matrix(r2, c2) ->
            Constraints.recordConstraint ctx env c1 r2 line
            let c1' = Constraints.resolveDim ctx c1
            let r2' = Constraints.resolveDim ctx r2
            if dimsDefinitelyConflict c1' r2' then
                let suggest =
                    not (dimsDefinitelyConflict r1 r2) &&
                    not (dimsDefinitelyConflict c1 c2)
                warnings.Add(warnMatmulMismatch line leftExpr rightExpr left right suggest)
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
            let r1' = Constraints.resolveDim ctx r1
            let r2' = Constraints.resolveDim ctx r2
            let c1' = Constraints.resolveDim ctx c1
            let c2' = Constraints.resolveDim ctx c2
            let rConflict = dimsDefinitelyConflict r1' r2'
            let cConflict = dimsDefinitelyConflict c1' c2'
            if rConflict || cConflict then
                warnings.Add(warnElementwiseMismatch line op leftExpr rightExpr left right)
                UnknownShape
            else Matrix(joinDim r1' r2', joinDim c1' c2')
        | _ -> UnknownShape

    else UnknownShape
