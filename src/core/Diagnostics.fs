// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Diagnostic record definitions and the tier classification that decides
// which warning codes appear in default mode versus --strict mode. Also
// holds the call stack context used for cross-function error traces.

module Diagnostics

open Ir
open Shapes
open WarningCodes

// --- Warning tier classification ---

/// Informational/noisy codes suppressed in default mode; shown with --strict.
let STRICT_ONLY_CODES : Set<WarningCode> =
    Set.ofList [
        // Parser / recognition limits
        W_UNSUPPORTED_STMT
        // Informational (not bugs)
        W_REASSIGN_INCOMPATIBLE
        W_RECURSIVE_LAMBDA; W_LAMBDA_CALL_APPROXIMATE
        W_MULTI_ASSIGN_NON_CALL
        // Stylistic / suspicious
        W_SUSPICIOUS_COMPARISON; W_MATRIX_COMPARISON
        W_LOGICAL_OP_NON_SCALAR; W_STRING_ARITHMETIC
        // 2D shape system limitation
        W_TOO_MANY_INDICES
        // cellfun/arrayfun output uniformity
        W_CELLFUN_NON_UNIFORM
    ]

/// Coder-mode-only warning codes (emitted only when --coder is active).
let CODER_ONLY_CODES : Set<WarningCode> =
    Set.ofList [
        W_CODER_VARIABLE_SIZE
        W_CODER_CELL_ARRAY
        W_CODER_DYNAMIC_FIELD
        W_CODER_TRY_CATCH
        W_CODER_UNSUPPORTED_BUILTIN
        W_CODER_RECURSION
    ]

// ---------------------------------------------------------------------------
// Diagnostic record
// ---------------------------------------------------------------------------

type Diagnostic = {
    line:       int
    code:       WarningCode
    message:    string
    relatedLine: int option
    col:        int
    relatedCol: int option
    callStack:  (string * int) list  // (funcName, callLine) pairs, innermost first
}

let diagnosticToString (d: Diagnostic) : string =
    let cs = codeString d.code
    // Special case: W_UNSUPPORTED_STMT uses line= format
    let mainLine =
        match d.code with
        | W_UNSUPPORTED_STMT -> $"{cs} line={d.line} {d.message}"
        | _                  -> $"{cs} line {d.line}: {d.message}"
    if d.callStack.IsEmpty then
        mainLine
    else
        let frames =
            d.callStack
            |> List.mapi (fun i (funcName, callLine) ->
                let indent = String.replicate (i + 1) "  "
                $"{indent}in {funcName}, called from line {callLine}")
        mainLine + "\n" + (frames |> String.concat "\n")

// ---------------------------------------------------------------------------
// Pretty-printing helpers for IR expressions
// ---------------------------------------------------------------------------

let rec prettyExprIr (expr: Expr) : string =
    match expr with
    | Var(_, name)        -> name
    | Const(_, v)         ->
        if v = System.Math.Floor(v : float) && not (System.Double.IsInfinity v) then
            string (int64 v)
        else
            string v
    | StringLit(_, s)     -> $"'{s}'"
    | Neg(_, operand)     -> $"(-{prettyExprIr operand})"
    | Not(_, operand)     -> $"(~{prettyExprIr operand})"
    | Transpose(_, operand) -> $"{prettyExprIr operand}'"
    | Lambda(_, parms, _) ->
        let ps = parms |> String.concat ", "
        $"@({ps}) <body>"
    | FuncHandle(_, name) -> $"@{name}"
    | Apply(_, base_, args) ->
        let baseName =
            match base_ with
            | Var(_, n) -> n
            | _         -> "<expr>"
        let argsStr = args |> List.map prettyIndexArgIr |> String.concat ", "
        $"{baseName}({argsStr})"
    | CurlyApply(_, base_, args) ->
        let baseName =
            match base_ with
            | Var(_, n) -> n
            | _         -> "<expr>"
        let argsStr = args |> List.map prettyIndexArgIr |> String.concat ", "
        $"{baseName}{{{argsStr}}}"
    | MatrixLit _            -> "[matrix]"
    | CellLit _              -> "{cell}"
    | BinOp(_, op, left, right) ->
        if op = ":" then
            $"{prettyExprIr left}:{prettyExprIr right}"
        else
            $"({prettyExprIr left} {op} {prettyExprIr right})"
    | FieldAccess(_, base_, field) ->
        $"{prettyExprIr base_}.{field}"
    | End _ -> "end"
    | MetaClass(_, name) -> $"?{name}"

and prettyIndexArgIr (arg: IndexArg) : string =
    match arg with
    | Colon _                    -> ":"
    | Ir.Range(_, s, e)          -> $"{prettyExprIr s}:{prettyExprIr e}"
    | Ir.SteppedRange(_, s, t, e) -> $"{prettyExprIr s}:{prettyExprIr t}:{prettyExprIr e}"
    | IndexExpr(_, e)            -> prettyExprIr e

// ---------------------------------------------------------------------------
// Warning message builders
// ---------------------------------------------------------------------------

let makeDiag line code message =
    { line = line; code = code; message = message; relatedLine = None; col = 0; relatedCol = None; callStack = [] }

let makeDiagRel line code message relLine =
    { line = line; code = code; message = message; relatedLine = Some relLine; col = 0; relatedCol = None; callStack = [] }

let hasUnsupported (diags: Diagnostic list) : bool =
    diags |> List.exists (fun d ->
        match d.code with
        | W_UNSUPPORTED_STMT -> true
        | _ -> false)

let warnReassignIncompatible (line: int) (name: string) (newShape: Shape) (oldShape: Shape) : Diagnostic =
    makeDiag line W_REASSIGN_INCOMPATIBLE
        $"'{name}' reassigned: {shapeToString newShape}, was {shapeToString oldShape}"

let warnSuspiciousComparisonMatrixScalar
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line W_SUSPICIOUS_COMPARISON
        $"{prettyExprIr leftExpr} {op} {prettyExprIr rightExpr}: matrix vs scalar, returns logical matrix"

let warnMatrixToMatrixComparison
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line W_MATRIX_COMPARISON
        $"{prettyExprIr leftExpr} {op} {prettyExprIr rightExpr}: matrix vs matrix, returns logical matrix"

let warnLogicalOpNonScalar
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line W_LOGICAL_OP_NON_SCALAR
        $"{prettyExprIr leftExpr} {op} {prettyExprIr rightExpr}: non-scalar operand(s) ({shapeToString left} vs {shapeToString right})"

let warnTooManyIndices (line: int) (expr: Expr) : Diagnostic =
    makeDiag line W_TOO_MANY_INDICES
        $"{prettyExprIr expr}: too many indices for 2D matrix. Shape assumed unknown."

let warnRangeEndpointsMustBeScalar (line: int) (arg: IndexArg) (startShape: Shape) (endShape: Shape) : Diagnostic =
    makeDiag line W_RANGE_NON_SCALAR
        $"{prettyIndexArgIr arg}: non-scalar range endpoints {shapeToString startShape}, {shapeToString endShape}. Shape assumed unknown."

let warnInvalidRangeEndLtStart (line: int) (arg: IndexArg) : Diagnostic =
    makeDiag line W_INVALID_RANGE
        $"{prettyIndexArgIr arg}: end < start"

let warnNonScalarIndexArg (line: int) (arg: IndexArg) (shape: Shape) : Diagnostic =
    makeDiag line W_NON_SCALAR_INDEX
        $"{prettyIndexArgIr arg}: non-scalar index {shapeToString shape}. Shape assumed unknown."

let warnElementwiseMismatch
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line W_ELEMENTWISE_MISMATCH
        $"{prettyExprIr leftExpr} {op} {prettyExprIr rightExpr}: {shapeToString left} vs {shapeToString right}"

let warnMatmulMismatch
    (line: int) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) (suggestElementwise: bool) : Diagnostic =
    let leftCols =
        match left with MatrixCols c -> dimStr c | _ -> "?"
    let rightRows =
        match right with MatrixRows r -> dimStr r | _ -> "?"
    let baseMsg =
        $"{prettyExprIr leftExpr} * {prettyExprIr rightExpr}: inner dims {leftCols} vs {rightRows}"
    let msg =
        if suggestElementwise then baseMsg + ". Use .* for elementwise"
        else baseMsg
    makeDiag line W_INNER_DIM_MISMATCH msg

let warnUnsupportedStmt (line: int) (raw: string) (targets: string list) : Diagnostic =
    let targetStr = if targets.IsEmpty then "(none)" else targets |> String.concat ", "
    let rawStr    = if raw = "" then "" else $" '{raw}'"
    makeDiag line W_UNSUPPORTED_STMT $"targets={targetStr}{rawStr}"

let warnUnknownFunction (line: int) (name: string) : Diagnostic =
    makeDiag line W_UNKNOWN_FUNCTION
        $"'{name}': unrecognized function. Shape assumed unknown."

let warnFunctionArgCountMismatch (line: int) (funcName: string) (expected: int) (got: int) : Diagnostic =
    makeDiag line W_FUNCTION_ARG_COUNT_MISMATCH
        $"{funcName}: expected {expected} args, {got} given"

let warnRecursiveFunction (line: int) (funcName: string) : Diagnostic =
    makeDiag line W_RECURSIVE_FUNCTION
        $"{funcName}: recursive call. Shape assumed unknown."

let warnProcedureInExpr (line: int) (funcName: string) : Diagnostic =
    makeDiag line W_PROCEDURE_IN_EXPR
        $"{funcName}: no return value, cannot use in expression"

let warnMultiAssignNonCall (line: int) : Diagnostic =
    makeDiag line W_MULTI_ASSIGN_NON_CALL
        "Destructuring requires function call on right side"

let warnMultiAssignCountMismatch (line: int) (funcName: string) (expected: int) (got: int) : Diagnostic =
    makeDiag line W_MULTI_ASSIGN_COUNT_MISMATCH
        $"{funcName}: returns {expected} values, {got} targets"

let warnMultiReturnCount (line: int) (fname: string) (supported: string) (got: int) : Diagnostic =
    makeDiag line W_MULTI_ASSIGN_COUNT_MISMATCH
        $"{fname}: supports {supported} return values, {got} given"

let warnStringArithmetic (line: int) (op: string) (leftShape: Shape) (rightShape: Shape) : Diagnostic =
    makeDiag line W_STRING_ARITHMETIC
        $"String arithmetic: {shapeToString leftShape} {op} {shapeToString rightShape}"

let warnStructFieldNotFound (line: int) (field: string) (structShape: Shape) : Diagnostic =
    makeDiag line W_STRUCT_FIELD_NOT_FOUND
        $"Field '{field}' not found in {shapeToString structShape}"

let warnFieldAccessNonStruct (line: int) (baseShape: Shape) : Diagnostic =
    makeDiag line W_FIELD_ACCESS_NON_STRUCT
        $"Field access on non-struct {shapeToString baseShape}"

let warnCurlyIndexingNonCell (line: int) (baseShape: Shape) : Diagnostic =
    makeDiag line W_CURLY_INDEXING_NON_CELL
        $"Curly-brace indexing on non-cell {shapeToString baseShape}"

let warnCellAssignNonCell (line: int) (baseName: string) (baseShape: Shape) : Diagnostic =
    makeDiag line W_CELL_ASSIGN_NON_CELL
        $"'{baseName}': cell assignment to non-cell {shapeToString baseShape}"

let warnIndexAssignTypeMismatch (line: int) (baseName: string) (baseShape: Shape) : Diagnostic =
    makeDiag line W_INDEX_ASSIGN_TYPE_MISMATCH
        $"'{baseName}': indexed assignment to non-indexable {shapeToString baseShape}"

let warnReturnOutsideFunction (line: int) : Diagnostic =
    makeDiag line W_RETURN_OUTSIDE_FUNCTION
        "return outside function"

let warnBreakOutsideLoop (line: int) : Diagnostic =
    makeDiag line W_BREAK_OUTSIDE_LOOP
        "break outside loop"

let warnContinueOutsideLoop (line: int) : Diagnostic =
    makeDiag line W_CONTINUE_OUTSIDE_LOOP
        "continue outside loop"

let warnLambdaCallApproximate (line: int) (varName: string) : Diagnostic =
    makeDiag line W_LAMBDA_CALL_APPROXIMATE
        $"Handle '{varName}': limited analysis. Shape assumed unknown."

let warnLambdaArgCountMismatch (line: int) (expected: int) (got: int) : Diagnostic =
    makeDiag line W_LAMBDA_ARG_COUNT_MISMATCH
        $"Lambda: expected {expected} args, {got} given"

let warnRecursiveLambda (line: int) : Diagnostic =
    makeDiag line W_RECURSIVE_LAMBDA
        "Recursive lambda. Shape assumed unknown."

let warnEndOutsideIndexing (line: int) : Diagnostic =
    makeDiag line W_END_OUTSIDE_INDEXING
        "'end' only valid inside indexing"

let warnExternalParseError (line: int) (fname: string) (sourcePath: string) : Diagnostic =
    makeDiag line W_EXTERNAL_PARSE_ERROR
        $"{fname}: parse error in {sourcePath}. Shape assumed unknown."

let warnConstraintConflict (line: int) (varName: string) (value: int) (otherDim: string) (sourceLine: int) : Diagnostic =
    makeDiag line W_CONSTRAINT_CONFLICT
        $"{varName}={value} conflicts with {varName}=={otherDim} (line {sourceLine})"

let warnReshapeMismatch (line: int) (inputShape: Shape) (m: string) (n: string) : Diagnostic =
    makeDiag line W_RESHAPE_MISMATCH
        $"reshape: {shapeToString inputShape} cannot fill {m}x{n}"

let warnDivisionByZero (line: int) (leftExpr: Expr) (rightExpr: Expr) : Diagnostic =
    ignore leftExpr
    ignore rightExpr
    makeDiag line W_DIVISION_BY_ZERO
        "Division by zero"

let warnIndexOutOfBounds (line: int) (indexVal: string) (dimSize: string) (definite: bool) : Diagnostic =
    let verb = if definite then "exceeds" else "may exceed"
    makeDiag line W_INDEX_OUT_OF_BOUNDS
        $"Index {indexVal} {verb} dimension {dimSize}"

let warnPossiblyNegativeDim (line: int) (dimVal: string) : Diagnostic =
    makeDiag line W_POSSIBLY_NEGATIVE_DIM
        $"Non-positive dimension: {dimVal}"

let warnArithmeticTypeMismatch
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    ignore leftExpr
    ignore rightExpr
    makeDiag line W_ARITHMETIC_TYPE_MISMATCH
        $"{op}: non-numeric operands {shapeToString left}, {shapeToString right}"

let warnTransposeTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line W_TRANSPOSE_TYPE_MISMATCH
        $"Transpose: non-numeric operand {shapeToString shape}"

let warnNegateTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line W_NEGATE_TYPE_MISMATCH
        $"Negation: non-numeric operand {shapeToString shape}"

let warnNotTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line W_NOT_TYPE_MISMATCH
        $"~: non-numeric operand {shapeToString shape}"

let warnConcatTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line W_CONCAT_TYPE_MISMATCH
        $"Concatenation: non-numeric element {shapeToString shape}"

let warnMldivideDimMismatch
    (line: int) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    let leftRows  = match left  with MatrixRows r -> dimStr r | _ -> "?"
    let rightRows = match right with MatrixRows r -> dimStr r | _ -> "?"
    makeDiag line W_MLDIVIDE_DIM_MISMATCH
        $"{prettyExprIr leftExpr} \\ {prettyExprIr rightExpr}: {leftRows} rows vs {rightRows} rows"

let warnMatrixPowerNonSquare (line: int) (expr: Expr) (shape: Shape) : Diagnostic =
    let rows = match shape with MatrixRows r -> dimStr r | _ -> "?"
    let cols = match shape with MatrixCols c -> dimStr c | _ -> "?"
    makeDiag line W_MATRIX_POWER_NON_SQUARE
        $"{prettyExprIr expr}^n: non-square {shapeToString shape} ({rows}x{cols})"

let warnHorzcatRowMismatch (line: int) (rowA: Dim) (rowB: Dim) : Diagnostic =
    makeDiag line W_HORZCAT_ROW_MISMATCH
        $"horzcat: row count {dimStr rowA} vs {dimStr rowB}"

let warnVertcatColMismatch (line: int) (colA: Dim) (colB: Dim) : Diagnostic =
    makeDiag line W_VERTCAT_COL_MISMATCH
        $"vertcat: col count {dimStr colA} vs {dimStr colB}"

let warnCellfunNonUniform (line: int) : Diagnostic =
    makeDiag line W_CELLFUN_NON_UNIFORM
        "cellfun: non-scalar output, use 'UniformOutput',false"

// ---------------------------------------------------------------------------
// Coder-mode warning builders (W_CODER_* family)
// ---------------------------------------------------------------------------

let warnCoderVariableSize (line: int) (varName: string) (shape: Shape) : Diagnostic =
    makeDiag line W_CODER_VARIABLE_SIZE
        $"Variable '{varName}' has unbounded dimension {shapeToString shape} (MATLAB Coder requires fixed-size or coder.varsize declaration)"

let warnCoderCellArray (line: int) (varName: string) : Diagnostic =
    makeDiag line W_CODER_CELL_ARRAY
        $"Variable '{varName}' is a cell array (limited support in MATLAB Coder)"

let warnCoderDynamicField (line: int) : Diagnostic =
    makeDiag line W_CODER_DYNAMIC_FIELD
        "Dynamic struct field access s.(expr) is not supported by MATLAB Coder"

let warnCoderTryCatch (line: int) : Diagnostic =
    makeDiag line W_CODER_TRY_CATCH
        "try/catch is not supported by MATLAB Coder"

let warnCoderUnsupportedBuiltin (line: int) (fname: string) : Diagnostic =
    makeDiag line W_CODER_UNSUPPORTED_BUILTIN
        $"Builtin '{fname}' is not supported by MATLAB Coder"

let warnCoderRecursion (line: int) (fname: string) : Diagnostic =
    makeDiag line W_CODER_RECURSION
        $"Recursive call to '{fname}' (MATLAB Coder supports limited recursion)"
