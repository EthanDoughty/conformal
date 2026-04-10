// Conformal Migrate: MATLAB-to-Python Transpiler
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Python AST definitions used as the target language for translation.
// Kept deliberately small: just enough node types to express what
// MATLAB source can be translated into, not a full CPython AST.

module PyAst

type PyExpr =
    | PyVar       of string
    | PyConst     of float
    | PyStr       of string
    | PyBool      of bool
    | PyNone
    | PyBinOp     of op: string * left: PyExpr * right: PyExpr
    | PyUnaryOp   of op: string * operand: PyExpr
    | PyCall      of func: PyExpr * args: PyExpr list * kwargs: (string * PyExpr) list
    | PyIndex     of base_: PyExpr * indices: PyIdx list
    | PyAttr      of base_: PyExpr * attr: string
    | PyList      of PyExpr list
    | PyTuple     of PyExpr list
    | PyArray     of PyExpr list list
    | PyLambda    of parms: string list * body: PyExpr
    | PyComment   of string

and PyIdx =
    | PyScalarIdx of PyExpr
    | PySlice     of lo: PyExpr option * hi: PyExpr option * step: PyExpr option

type PyStmt =
    | PyAssign      of target: string * expr: PyExpr
    | PyMultiAssign of targets: string list * expr: PyExpr
    | PyExprStmt    of PyExpr
    | PyIf          of cond: PyExpr * thenBody: PyStmt list
                       * elifs: (PyExpr * PyStmt list) list * elseBody: PyStmt list
    | PyFor         of var_: string * iterable: PyExpr * body: PyStmt list
    | PyWhile       of cond: PyExpr * body: PyStmt list
    | PyFuncDef     of name: string * parms: string list * body: PyStmt list
                       * returnVars: string list
    | PyReturn      of PyExpr list
    | PyCommentStmt of string
    | PyImport      of module_: string * alias: string option
    | PyFromImport  of module_: string * names: string list
    | PyPass
    | PyBreak
    | PyContinue
    | PyTry         of tryBody: PyStmt list * exceptBody: PyStmt list

type PyProgram = { imports: PyStmt list; body: PyStmt list }
