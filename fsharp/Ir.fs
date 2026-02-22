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

    member this.Line =
        match this with
        | Var(l,_,_) | Const(l,_,_) | StringLit(l,_,_)
        | Neg(l,_,_) | Not(l,_,_) | BinOp(l,_,_,_,_)
        | Transpose(l,_,_) | FieldAccess(l,_,_,_)
        | Lambda(l,_,_,_) | FuncHandle(l,_,_) | End(l,_)
        | Apply(l,_,_,_) | CurlyApply(l,_,_,_)
        | MatrixLit(l,_,_) | CellLit(l,_,_) -> l

    member this.Col =
        match this with
        | Var(_,c,_) | Const(_,c,_) | StringLit(_,c,_)
        | Neg(_,c,_) | Not(_,c,_) | BinOp(_,c,_,_,_)
        | Transpose(_,c,_) | FieldAccess(_,c,_,_)
        | Lambda(_,c,_,_) | FuncHandle(_,c,_) | End(_,c)
        | Apply(_,c,_,_) | CurlyApply(_,c,_,_)
        | MatrixLit(_,c,_) | CellLit(_,c,_) -> c

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

    member this.Line =
        match this with
        | Assign(l,_,_,_) | StructAssign(l,_,_,_,_) | CellAssign(l,_,_,_,_)
        | IndexAssign(l,_,_,_,_) | IndexStructAssign(l,_,_,_,_,_,_)
        | FieldIndexAssign(l,_,_,_,_,_,_,_) | ExprStmt(l,_,_)
        | If(l,_,_,_,_) | IfChain(l,_,_,_,_) | While(l,_,_,_) | For(l,_,_,_,_)
        | Switch(l,_,_,_,_) | Try(l,_,_,_) | Break(l,_) | Continue(l,_) | Return(l,_)
        | OpaqueStmt(l,_,_,_) | FunctionDef(l,_,_,_,_,_) | AssignMulti(l,_,_,_) -> l

    member this.Col =
        match this with
        | Assign(_,c,_,_) | StructAssign(_,c,_,_,_) | CellAssign(_,c,_,_,_)
        | IndexAssign(_,c,_,_,_) | IndexStructAssign(_,c,_,_,_,_,_)
        | FieldIndexAssign(_,c,_,_,_,_,_,_) | ExprStmt(_,c,_)
        | If(_,c,_,_,_) | IfChain(_,c,_,_,_) | While(_,c,_,_) | For(_,c,_,_,_)
        | Switch(_,c,_,_,_) | Try(_,c,_,_) | Break(_,c) | Continue(_,c) | Return(_,c)
        | OpaqueStmt(_,c,_,_) | FunctionDef(_,c,_,_,_,_) | AssignMulti(_,c,_,_) -> c

type Program = { body: Stmt list }
