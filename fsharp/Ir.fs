module Ir

// Intermediate Representation for MATLAB.
// Mirrors ir/ir.py using F# discriminated unions.
// camelCase for record fields, PascalCase for DU cases and type names.

type Expr =
    | Var         of line: int * col: int * name: string
    | Const       of line: int * col: int * value: float
    | StringLit   of line: int * col: int * value: string
    | Neg         of line: int * col: int * operand: Expr
    | Not         of line: int * col: int * operand: Expr
    | BinOp       of line: int * col: int * op: string * left: Expr * right: Expr
    | Transpose   of line: int * col: int * operand: Expr
    | FieldAccess of line: int * col: int * base_: Expr * field: string
    | Lambda      of line: int * col: int * parms: string list * body: Expr
    | FuncHandle  of line: int * col: int * name: string
    | End         of line: int * col: int
    | Apply       of line: int * col: int * base_: Expr * args: IndexArg list
    | CurlyApply  of line: int * col: int * base_: Expr * args: IndexArg list
    | MatrixLit   of line: int * col: int * rows: Expr list list
    | CellLit     of line: int * col: int * rows: Expr list list

and IndexArg =
    | Colon     of line: int * col: int
    | Range     of line: int * col: int * start: Expr * end_: Expr
    | IndexExpr of line: int * col: int * expr: Expr

type Stmt =
    | Assign           of line: int * col: int * name: string * expr: Expr
    | StructAssign     of line: int * col: int * baseName: string * fields: string list * expr: Expr
    | CellAssign       of line: int * col: int * baseName: string * args: IndexArg list * expr: Expr
    | IndexAssign      of line: int * col: int * baseName: string * args: IndexArg list * expr: Expr
    | IndexStructAssign of line: int * col: int * baseName: string * indexArgs: IndexArg list
                          * indexKind: string * fields: string list * expr: Expr
    | FieldIndexAssign of line: int * col: int * baseName: string * prefixFields: string list
                          * indexArgs: IndexArg list * indexKind: string * suffixFields: string list * expr: Expr
    | ExprStmt         of line: int * col: int * expr: Expr
    | If               of line: int * col: int * cond: Expr * thenBody: Stmt list * elseBody: Stmt list
    | IfChain          of line: int * col: int * conditions: Expr list * bodies: Stmt list list * elseBody: Stmt list
    | While            of line: int * col: int * cond: Expr * body: Stmt list
    | For              of line: int * col: int * var_: string * it: Expr * body: Stmt list
    | Switch           of line: int * col: int * expr: Expr * cases: (Expr * Stmt list) list * otherwise: Stmt list
    | Try              of line: int * col: int * tryBody: Stmt list * catchBody: Stmt list
    | Break            of line: int * col: int
    | Continue         of line: int * col: int
    | Return           of line: int * col: int
    | OpaqueStmt       of line: int * col: int * targets: string list * raw: string
    | FunctionDef      of line: int * col: int * name: string * parms: string list
                          * outputVars: string list * body: Stmt list
    | AssignMulti      of line: int * col: int * targets: string list * expr: Expr

type Program = { body: Stmt list }
