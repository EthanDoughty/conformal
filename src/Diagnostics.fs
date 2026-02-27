module Diagnostics

open Ir
open Shapes

// ---------------------------------------------------------------------------
// Warning tier classification
// Codes suppressed in default mode; shown only with --strict.
// ---------------------------------------------------------------------------

let STRICT_ONLY_CODES : Set<string> =
    Set.ofList [
        // Parser / recognition limits
        "W_UNSUPPORTED_STMT"; "W_UNSUPPORTED_MULTI_ASSIGN"
        "W_UNKNOWN_FUNCTION"; "W_EXTERNAL_PARSE_ERROR"
        // Cascade-prone field / cell warnings
        "W_STRUCT_FIELD_NOT_FOUND"; "W_FIELD_ACCESS_NON_STRUCT"
        "W_CURLY_INDEXING_NON_CELL"; "W_CELL_ASSIGN_NON_CELL"
        // Informational (not bugs)
        "W_REASSIGN_INCOMPATIBLE"; "W_RECURSIVE_FUNCTION"
        "W_RECURSIVE_LAMBDA"; "W_LAMBDA_CALL_APPROXIMATE"
        "W_MULTI_ASSIGN_NON_CALL"; "W_MULTI_ASSIGN_BUILTIN"
        // Stylistic / suspicious
        "W_SUSPICIOUS_COMPARISON"; "W_MATRIX_COMPARISON"
        "W_LOGICAL_OP_NON_SCALAR"; "W_STRING_ARITHMETIC"
        // 2D shape system limitation
        "W_TOO_MANY_INDICES"
        // cellfun/arrayfun output uniformity
        "W_CELLFUN_NON_UNIFORM"
    ]

/// Coder-mode-only warning codes (emitted only when --coder is active).
let CODER_ONLY_CODES : Set<string> =
    Set.ofList [
        "W_CODER_VARIABLE_SIZE"
        "W_CODER_CELL_ARRAY"
        "W_CODER_DYNAMIC_FIELD"
        "W_CODER_TRY_CATCH"
        "W_CODER_UNSUPPORTED_BUILTIN"
        "W_CODER_RECURSION"
    ]

// ---------------------------------------------------------------------------
// Diagnostic record
// ---------------------------------------------------------------------------

type Diagnostic = {
    line:       int
    code:       string   // W_* prefix
    message:    string
    relatedLine: int option
    col:        int
    relatedCol: int option
}

let diagnosticToString (d: Diagnostic) : string =
    // Special case: W_UNSUPPORTED_STMT uses line= format
    if d.code = "W_UNSUPPORTED_STMT" then
        d.code + " line=" + string d.line + " " + d.message
    elif d.code <> "" then
        d.code + " line " + string d.line + ": " + d.message
    else
        "Line " + string d.line + ": " + d.message

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
    | StringLit(_, s)     -> "'" + s + "'"
    | Neg(_, operand)     -> "(-" + prettyExprIr operand + ")"
    | Not(_, operand)     -> "(~" + prettyExprIr operand + ")"
    | Transpose(_, operand) -> prettyExprIr operand + "'"
    | Lambda(_, parms, _) ->
        let ps = parms |> String.concat ", "
        "@(" + ps + ") <body>"
    | FuncHandle(_, name) -> "@" + name
    | Apply(_, base_, args) ->
        let baseName =
            match base_ with
            | Var(_, n) -> n
            | _         -> "<expr>"
        let argsStr = args |> List.map prettyIndexArgIr |> String.concat ", "
        baseName + "(" + argsStr + ")"
    | CurlyApply(_, base_, args) ->
        let baseName =
            match base_ with
            | Var(_, n) -> n
            | _         -> "<expr>"
        let argsStr = args |> List.map prettyIndexArgIr |> String.concat ", "
        baseName + "{" + argsStr + "}"
    | MatrixLit _            -> "[matrix]"
    | CellLit _              -> "{cell}"
    | BinOp(_, op, left, right) ->
        if op = ":" then
            prettyExprIr left + ":" + prettyExprIr right
        else
            "(" + prettyExprIr left + " " + op + " " + prettyExprIr right + ")"
    | FieldAccess(_, base_, field) ->
        prettyExprIr base_ + "." + field
    | End _ -> "end"

and prettyIndexArgIr (arg: IndexArg) : string =
    match arg with
    | Colon _              -> ":"
    | Ir.Range(_, s, e)    -> prettyExprIr s + ":" + prettyExprIr e
    | IndexExpr(_, e)      -> prettyExprIr e

// ---------------------------------------------------------------------------
// Warning message builders
// ---------------------------------------------------------------------------

let makeDiag line code message =
    { line = line; code = code; message = message; relatedLine = None; col = 0; relatedCol = None }

let makeDiagRel line code message relLine =
    { line = line; code = code; message = message; relatedLine = Some relLine; col = 0; relatedCol = None }

let hasUnsupported (diags: Diagnostic list) : bool =
    diags |> List.exists (fun d -> d.code.StartsWith("W_UNSUPPORTED_"))

let warnReassignIncompatible (line: int) (name: string) (newShape: Shape) (oldShape: Shape) : Diagnostic =
    makeDiag line "W_REASSIGN_INCOMPATIBLE"
        ("Variable '" + name + "' reassigned with incompatible shape " + shapeToString newShape +
         " (previously " + shapeToString oldShape + ")")

let warnSuspiciousComparisonMatrixScalar
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line "W_SUSPICIOUS_COMPARISON"
        ("Suspicious comparison between matrix and scalar in (" +
         prettyExprIr leftExpr + " " + op + " " + prettyExprIr rightExpr +
         ") (" + shapeToString left + " vs " + shapeToString right +
         "). In MATLAB this is elementwise and may produce a logical matrix.")

let warnMatrixToMatrixComparison
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line "W_MATRIX_COMPARISON"
        ("Matrix-to-matrix comparison in (" +
         prettyExprIr leftExpr + " " + op + " " + prettyExprIr rightExpr +
         ") (" + shapeToString left + " vs " + shapeToString right +
         "). In MATLAB this is elementwise and may produce a logical matrix.")

let warnLogicalOpNonScalar
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line "W_LOGICAL_OP_NON_SCALAR"
        ("Logical operator " + op + " used with non-scalar operand(s) in (" +
         prettyExprIr leftExpr + " " + op + " " + prettyExprIr rightExpr +
         ") (" + shapeToString left + " vs " + shapeToString right + ").")

let warnIndexingScalar (line: int) (expr: Expr) : Diagnostic =
    makeDiag line "W_INDEXING_SCALAR"
        ("Indexing applied to scalar in " + prettyExprIr expr + ". Treating result as unknown.")

let warnTooManyIndices (line: int) (expr: Expr) : Diagnostic =
    makeDiag line "W_TOO_MANY_INDICES"
        ("Too many indices for 2D matrix in " + prettyExprIr expr + ". Treating result as unknown.")

let warnRangeEndpointsMustBeScalar (line: int) (arg: IndexArg) (startShape: Shape) (endShape: Shape) : Diagnostic =
    makeDiag line "W_RANGE_NON_SCALAR"
        ("Range endpoints in indexing must be scalar; got " +
         shapeToString startShape + " and " + shapeToString endShape +
         " in " + prettyIndexArgIr arg + ". Treating result as unknown.")

let warnInvalidRangeEndLtStart (line: int) (arg: IndexArg) : Diagnostic =
    makeDiag line "W_INVALID_RANGE"
        ("Invalid range in indexing (" + prettyIndexArgIr arg + "): end < start.")

let warnNonScalarIndexArg (line: int) (arg: IndexArg) (shape: Shape) : Diagnostic =
    makeDiag line "W_NON_SCALAR_INDEX"
        ("Non-scalar index argument " + prettyIndexArgIr arg + " has shape " + shapeToString shape +
         ". Treating indexing result as unknown.")

let warnElementwiseMismatch
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    makeDiag line "W_ELEMENTWISE_MISMATCH"
        ("Elementwise " + op + " mismatch in (" +
         prettyExprIr leftExpr + " " + op + " " + prettyExprIr rightExpr +
         "): " + shapeToString left + " vs " + shapeToString right)

let warnMatmulMismatch
    (line: int) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) (suggestElementwise: bool) : Diagnostic =
    let leftCols =
        match left with Matrix(_, c) -> dimStr c | _ -> "?"
    let rightRows =
        match right with Matrix(r, _) -> dimStr r | _ -> "?"
    let mutable msg =
        "Dimension mismatch in expression (" +
        prettyExprIr leftExpr + " * " + prettyExprIr rightExpr + "): inner dims " +
        leftCols + " vs " + rightRows +
        " (shapes " + shapeToString left + " and " + shapeToString right + ")"
    if suggestElementwise then
        msg <- msg + ". Did you mean elementwise multiplication (.*)?"
    makeDiag line "W_INNER_DIM_MISMATCH" msg

let warnUnsupportedStmt (line: int) (raw: string) (targets: string list) : Diagnostic =
    let targetStr = if targets.IsEmpty then "(none)" else targets |> String.concat ", "
    let rawStr    = if raw = "" then "" else " '" + raw + "'"
    makeDiag line "W_UNSUPPORTED_STMT" ("targets=" + targetStr + rawStr)

let warnUnknownFunction (line: int) (name: string) : Diagnostic =
    makeDiag line "W_UNKNOWN_FUNCTION"
        ("Function '" + name + "' is not recognized; treating result as unknown")

let warnUnsupportedMultiAssign (line: int) : Diagnostic =
    makeDiag line "W_UNSUPPORTED_MULTI_ASSIGN"
        "Destructuring assignment not yet supported (Phase C)"

let warnFunctionArgCountMismatch (line: int) (funcName: string) (expected: int) (got: int) : Diagnostic =
    makeDiag line "W_FUNCTION_ARG_COUNT_MISMATCH"
        ("function " + funcName + " expects " + string expected + " arguments, got " + string got)

let warnRecursiveFunction (line: int) (funcName: string) : Diagnostic =
    makeDiag line "W_RECURSIVE_FUNCTION"
        ("recursive call to " + funcName + " not supported (returns unknown)")

let warnProcedureInExpr (line: int) (funcName: string) : Diagnostic =
    makeDiag line "W_PROCEDURE_IN_EXPR"
        ("procedure " + funcName + " has no return value, cannot be used in expression")

let warnMultiAssignNonCall (line: int) : Diagnostic =
    makeDiag line "W_MULTI_ASSIGN_NON_CALL"
        "destructuring assignment requires function call on RHS"

let warnMultiAssignBuiltin (line: int) (funcName: string) : Diagnostic =
    makeDiag line "W_MULTI_ASSIGN_BUILTIN"
        ("builtin " + funcName + " does not support multiple returns")

let warnMultiAssignCountMismatch (line: int) (funcName: string) (expected: int) (got: int) : Diagnostic =
    makeDiag line "W_MULTI_ASSIGN_COUNT_MISMATCH"
        ("function " + funcName + " returns " + string expected + " values, got " + string got + " targets")

let warnMultiReturnCount (line: int) (fname: string) (supported: string) (got: int) : Diagnostic =
    makeDiag line "W_MULTI_ASSIGN_COUNT_MISMATCH"
        ("builtin " + fname + " supports " + supported + " return values, got " + string got)

let warnStringArithmetic (line: int) (op: string) (leftShape: Shape) (rightShape: Shape) : Diagnostic =
    makeDiag line "W_STRING_ARITHMETIC"
        ("Invalid string arithmetic (" + shapeToString leftShape + " " + op + " " + shapeToString rightShape + ")")

let warnStructFieldNotFound (line: int) (field: string) (structShape: Shape) : Diagnostic =
    makeDiag line "W_STRUCT_FIELD_NOT_FOUND"
        ("Field '" + field + "' not found in " + shapeToString structShape)

let warnFieldAccessNonStruct (line: int) (baseShape: Shape) : Diagnostic =
    makeDiag line "W_FIELD_ACCESS_NON_STRUCT"
        ("Field access on non-struct value (" + shapeToString baseShape + ")")

let warnCurlyIndexingNonCell (line: int) (baseShape: Shape) : Diagnostic =
    makeDiag line "W_CURLY_INDEXING_NON_CELL"
        ("Curly indexing on non-cell value (" + shapeToString baseShape + ")")

let warnCellAssignNonCell (line: int) (baseName: string) (baseShape: Shape) : Diagnostic =
    makeDiag line "W_CELL_ASSIGN_NON_CELL"
        ("Cell assignment to non-cell variable '" + baseName + "' (" + shapeToString baseShape + ")")

let warnIndexAssignTypeMismatch (line: int) (baseName: string) (baseShape: Shape) : Diagnostic =
    makeDiag line "W_INDEX_ASSIGN_TYPE_MISMATCH"
        ("Indexed assignment to non-indexable variable '" + baseName + "' (" + shapeToString baseShape + ")")

let warnReturnOutsideFunction (line: int) : Diagnostic =
    makeDiag line "W_RETURN_OUTSIDE_FUNCTION"
        "return statement outside function body"

let warnBreakOutsideLoop (line: int) : Diagnostic =
    makeDiag line "W_BREAK_OUTSIDE_LOOP"
        "break statement outside loop (treated as no-op)"

let warnContinueOutsideLoop (line: int) : Diagnostic =
    makeDiag line "W_CONTINUE_OUTSIDE_LOOP"
        "continue statement outside loop (treated as no-op)"

let warnLambdaCallApproximate (line: int) (varName: string) : Diagnostic =
    makeDiag line "W_LAMBDA_CALL_APPROXIMATE"
        ("Calling function handle '" + varName + "' returns unknown (body analysis deferred to v0.12.1)")

let warnLambdaArgCountMismatch (line: int) (expected: int) (got: int) : Diagnostic =
    makeDiag line "W_LAMBDA_ARG_COUNT_MISMATCH"
        ("lambda expects " + string expected + " arguments, got " + string got)

let warnRecursiveLambda (line: int) : Diagnostic =
    makeDiag line "W_RECURSIVE_LAMBDA"
        "recursive lambda call not supported (returns unknown)"

let warnEndOutsideIndexing (line: int) : Diagnostic =
    makeDiag line "W_END_OUTSIDE_INDEXING"
        "'end' keyword only valid inside indexing expressions"

let warnExternalParseError (line: int) (fname: string) (sourcePath: string) : Diagnostic =
    makeDiag line "W_EXTERNAL_PARSE_ERROR"
        ("Cannot analyze " + fname + " (parse error in " + sourcePath + "); treating result as unknown")

let warnConstraintConflict (line: int) (varName: string) (value: int) (otherDim: string) (sourceLine: int) : Diagnostic =
    makeDiag line "W_CONSTRAINT_CONFLICT"
        (varName + "=" + string value + " conflicts with " + varName + "==" + otherDim + " (from line " + string sourceLine + ")")

let warnReshapeMismatch (line: int) (inputShape: Shape) (m: string) (n: string) : Diagnostic =
    makeDiag line "W_RESHAPE_MISMATCH"
        ("reshape changes element count: " + shapeToString inputShape +
         " has different element count than " + m + "x" + n)

let warnDivisionByZero (line: int) (leftExpr: Expr) (rightExpr: Expr) : Diagnostic =
    // leftExpr and rightExpr kept for future use / symmetry with Python
    ignore leftExpr
    ignore rightExpr
    makeDiag line "W_DIVISION_BY_ZERO"
        "division by zero: divisor is definitely zero"

let warnIndexOutOfBounds (line: int) (indexVal: string) (dimSize: string) (definite: bool) : Diagnostic =
    let verb = if definite then "exceeds" else "may exceed"
    makeDiag line "W_INDEX_OUT_OF_BOUNDS"
        ("index out of bounds: index range " + indexVal + " " + verb + " dimension " + dimSize)

let warnPossiblyNegativeDim (line: int) (dimVal: string) : Diagnostic =
    makeDiag line "W_POSSIBLY_NEGATIVE_DIM"
        ("non-positive dimension: " + dimVal)

let warnArithmeticTypeMismatch
    (line: int) (op: string) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    ignore leftExpr
    ignore rightExpr
    makeDiag line "W_ARITHMETIC_TYPE_MISMATCH"
        ("Arithmetic operator " + op + " requires numeric operands, got " +
         shapeToString left + " and " + shapeToString right)

let warnTransposeTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line "W_TRANSPOSE_TYPE_MISMATCH"
        ("Transpose requires numeric operand, got " + shapeToString shape)

let warnNegateTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line "W_NEGATE_TYPE_MISMATCH"
        ("Negation requires numeric operand, got " + shapeToString shape)

let warnNotTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line "W_NOT_TYPE_MISMATCH"
        ("Logical NOT requires numeric operand, got " + shapeToString shape)

let warnConcatTypeMismatch (line: int) (shape: Shape) : Diagnostic =
    makeDiag line "W_CONCAT_TYPE_MISMATCH"
        ("Concatenation requires numeric elements, got " + shapeToString shape)

let warnMldivideDimMismatch
    (line: int) (leftExpr: Expr) (rightExpr: Expr) (left: Shape) (right: Shape) : Diagnostic =
    let leftRows  = match left  with Matrix(r, _) -> dimStr r | _ -> "?"
    let rightRows = match right with Matrix(r, _) -> dimStr r | _ -> "?"
    makeDiag line "W_MLDIVIDE_DIM_MISMATCH"
        ("Dimension mismatch in mldivide (" +
         prettyExprIr leftExpr + " \\ " + prettyExprIr rightExpr +
         "): A has " + leftRows + " rows but b has " + rightRows +
         " rows (shapes " + shapeToString left + " and " + shapeToString right + ")")

let warnMatrixPowerNonSquare (line: int) (expr: Expr) (shape: Shape) : Diagnostic =
    let rows = match shape with Matrix(r, _) -> dimStr r | _ -> "?"
    let cols = match shape with Matrix(_, c) -> dimStr c | _ -> "?"
    makeDiag line "W_MATRIX_POWER_NON_SQUARE"
        ("Matrix power (^) requires square matrix; " +
         prettyExprIr expr + " has shape " + shapeToString shape +
         " (" + rows + " rows, " + cols + " cols)")

let warnHorzcatRowMismatch (line: int) (rowA: Dim) (rowB: Dim) : Diagnostic =
    makeDiag line "W_HORZCAT_ROW_MISMATCH"
        ("Horizontal concatenation row count mismatch: " + dimStr rowA + " vs " + dimStr rowB)

let warnVertcatColMismatch (line: int) (colA: Dim) (colB: Dim) : Diagnostic =
    makeDiag line "W_VERTCAT_COL_MISMATCH"
        ("Vertical concatenation column count mismatch: " + dimStr colA + " vs " + dimStr colB)

let warnCellfunNonUniform (line: int) : Diagnostic =
    makeDiag line "W_CELLFUN_NON_UNIFORM"
        "cellfun returns non-scalar elements; set 'UniformOutput',false to collect into a cell array"

// ---------------------------------------------------------------------------
// Coder-mode warning builders (W_CODER_* family)
// ---------------------------------------------------------------------------

let warnCoderVariableSize (line: int) (varName: string) (shape: Shape) : Diagnostic =
    makeDiag line "W_CODER_VARIABLE_SIZE"
        ("Variable '" + varName + "' has unbounded dimension " + shapeToString shape +
         " (MATLAB Coder requires fixed-size or coder.varsize declaration)")

let warnCoderCellArray (line: int) (varName: string) : Diagnostic =
    makeDiag line "W_CODER_CELL_ARRAY"
        ("Variable '" + varName + "' is a cell array (limited support in MATLAB Coder)")

let warnCoderDynamicField (line: int) : Diagnostic =
    makeDiag line "W_CODER_DYNAMIC_FIELD"
        "Dynamic struct field access s.(expr) is not supported by MATLAB Coder"

let warnCoderTryCatch (line: int) : Diagnostic =
    makeDiag line "W_CODER_TRY_CATCH"
        "try/catch is not supported by MATLAB Coder"

let warnCoderUnsupportedBuiltin (line: int) (fname: string) : Diagnostic =
    makeDiag line "W_CODER_UNSUPPORTED_BUILTIN"
        ("Builtin '" + fname + "' is not supported by MATLAB Coder")

let warnCoderRecursion (line: int) (fname: string) : Diagnostic =
    makeDiag line "W_CODER_RECURSION"
        ("Recursive call to '" + fname + "' (MATLAB Coder supports limited recursion)")
