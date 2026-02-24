module Ir

// Intermediate Representation for MATLAB.
// Mirrors ir/ir.py using F# discriminated unions.
// camelCase for record fields, PascalCase for DU cases and type names.

// Source location: line and column (1-based).
// Not a struct: Fable does not support [<Struct>] on records used in DU cases.
type SrcLoc = { line: int; col: int }

/// Convenience constructor for SrcLoc.
let loc (line: int) (col: int) : SrcLoc = { line = line; col = col }

type Expr =
    | Var         of loc: SrcLoc * name: string
    | Const       of loc: SrcLoc * value: float
    | StringLit   of loc: SrcLoc * value: string
    | Neg         of loc: SrcLoc * operand: Expr
    | Not         of loc: SrcLoc * operand: Expr
    | BinOp       of loc: SrcLoc * op: string * left: Expr * right: Expr
    | Transpose   of loc: SrcLoc * operand: Expr
    | FieldAccess of loc: SrcLoc * base_: Expr * field: string
    | Lambda      of loc: SrcLoc * parms: string list * body: Expr
    | FuncHandle  of loc: SrcLoc * name: string
    | End         of loc: SrcLoc
    | Apply       of loc: SrcLoc * base_: Expr * args: IndexArg list
    | CurlyApply  of loc: SrcLoc * base_: Expr * args: IndexArg list
    | MatrixLit   of loc: SrcLoc * rows: Expr list list
    | CellLit     of loc: SrcLoc * rows: Expr list list

    member this.Line =
        match this with
        | Var(l,_) | Const(l,_) | StringLit(l,_)
        | Neg(l,_) | Not(l,_) | BinOp(l,_,_,_)
        | Transpose(l,_) | FieldAccess(l,_,_)
        | Lambda(l,_,_) | FuncHandle(l,_) | End l
        | Apply(l,_,_) | CurlyApply(l,_,_)
        | MatrixLit(l,_) | CellLit(l,_) -> l.line

    member this.Col =
        match this with
        | Var(l,_) | Const(l,_) | StringLit(l,_)
        | Neg(l,_) | Not(l,_) | BinOp(l,_,_,_)
        | Transpose(l,_) | FieldAccess(l,_,_)
        | Lambda(l,_,_) | FuncHandle(l,_) | End l
        | Apply(l,_,_) | CurlyApply(l,_,_)
        | MatrixLit(l,_) | CellLit(l,_) -> l.col

and IndexArg =
    | Colon     of loc: SrcLoc
    | Range     of loc: SrcLoc * start: Expr * end_: Expr
    | IndexExpr of loc: SrcLoc * expr: Expr

type Stmt =
    | Assign           of loc: SrcLoc * name: string * expr: Expr
    | StructAssign     of loc: SrcLoc * baseName: string * fields: string list * expr: Expr
    | CellAssign       of loc: SrcLoc * baseName: string * args: IndexArg list * expr: Expr
    | IndexAssign      of loc: SrcLoc * baseName: string * args: IndexArg list * expr: Expr
    | IndexStructAssign of loc: SrcLoc * baseName: string * indexArgs: IndexArg list
                          * indexKind: string * fields: string list * expr: Expr
    | FieldIndexAssign of loc: SrcLoc * baseName: string * prefixFields: string list
                          * indexArgs: IndexArg list * indexKind: string * suffixFields: string list * expr: Expr
    | ExprStmt         of loc: SrcLoc * expr: Expr
    | If               of loc: SrcLoc * cond: Expr * thenBody: Stmt list * elseBody: Stmt list
    | IfChain          of loc: SrcLoc * conditions: Expr list * bodies: Stmt list list * elseBody: Stmt list
    | While            of loc: SrcLoc * cond: Expr * body: Stmt list
    | For              of loc: SrcLoc * var_: string * it: Expr * body: Stmt list
    | Switch           of loc: SrcLoc * expr: Expr * cases: (Expr * Stmt list) list * otherwise: Stmt list
    | Try              of loc: SrcLoc * tryBody: Stmt list * catchBody: Stmt list
    | Break            of loc: SrcLoc
    | Continue         of loc: SrcLoc
    | Return           of loc: SrcLoc
    | OpaqueStmt       of loc: SrcLoc * targets: string list * raw: string
    | FunctionDef      of loc: SrcLoc * name: string * parms: string list
                          * outputVars: string list * body: Stmt list
    | AssignMulti      of loc: SrcLoc * targets: string list * expr: Expr

    member this.Line =
        match this with
        | Assign(l,_,_) | StructAssign(l,_,_,_) | CellAssign(l,_,_,_)
        | IndexAssign(l,_,_,_) | IndexStructAssign(l,_,_,_,_,_)
        | FieldIndexAssign(l,_,_,_,_,_,_) | ExprStmt(l,_)
        | If(l,_,_,_) | IfChain(l,_,_,_) | While(l,_,_) | For(l,_,_,_)
        | Switch(l,_,_,_) | Try(l,_,_) | Break l | Continue l | Return l
        | OpaqueStmt(l,_,_) | FunctionDef(l,_,_,_,_) | AssignMulti(l,_,_) -> l.line

    member this.Col =
        match this with
        | Assign(l,_,_) | StructAssign(l,_,_,_) | CellAssign(l,_,_,_)
        | IndexAssign(l,_,_,_) | IndexStructAssign(l,_,_,_,_,_)
        | FieldIndexAssign(l,_,_,_,_,_,_) | ExprStmt(l,_)
        | If(l,_,_,_) | IfChain(l,_,_,_) | While(l,_,_) | For(l,_,_,_)
        | Switch(l,_,_,_) | Try(l,_,_) | Break l | Continue l | Return l
        | OpaqueStmt(l,_,_) | FunctionDef(l,_,_,_,_) | AssignMulti(l,_,_) -> l.col

type Program = { body: Stmt list }
