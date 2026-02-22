module Json

open System
open System.Text
open Ir

// ---------------------------------------------------------------------------
// Low-level JSON helpers (manual string building, 2-space indentation).
// Matches the output of Python's json.dumps(..., indent=2).
// ---------------------------------------------------------------------------

let private escape (s: string) : string =
    let sb = StringBuilder()
    for c in s do
        match c with
        | '"'  -> sb.Append("\\\"") |> ignore
        | '\\' -> sb.Append("\\\\") |> ignore
        | '\n' -> sb.Append("\\n")  |> ignore
        | '\r' -> sb.Append("\\r")  |> ignore
        | '\t' -> sb.Append("\\t")  |> ignore
        | c when int c < 32 ->
            sb.Append("\\u" + (int c).ToString("x4")) |> ignore
        | c -> sb.Append(c) |> ignore
    sb.ToString()

let private jStr (s: string) = "\"" + escape s + "\""

// Format a float the way Python's json module does:
//  - Uses shortest round-trip representation (Python's repr behavior)
//  - Integer-valued floats get ".0" suffix: 5.0 -> "5.0"
//  - Exponent marker is lowercase: 1e-10 not 1E-10
let private jFloat (v: float) : string =
    if Double.IsPositiveInfinity v then "1e308"  // large sentinel
    elif Double.IsNegativeInfinity v then "-1e308"
    elif Double.IsNaN v then "null"
    else
        // .NET "R" format gives shortest round-trip representation.
        let s = v.ToString("R")
        // Make exponent lowercase to match Python json.dumps
        let s = s.Replace("E+0", "e+0").Replace("E-0", "e-0")
                  .Replace("E+", "e+").Replace("E-", "e-")
                  .Replace("E", "e")
        // If no decimal point or exponent, add ".0" (Python keeps "5.0" not "5")
        if s.Contains('.') || s.Contains('e') || s.Contains('n') then s
        else s + ".0"

// ---------------------------------------------------------------------------
// Indented builder
// ---------------------------------------------------------------------------

// We build the JSON by emitting structured nodes and tracking indentation.
// This matches Python's json.dumps(obj, indent=2) output character-for-character.

type private Writer(sb: StringBuilder) =
    let mutable indent = 0
    let indentStr () = String.replicate indent " "

    member _.WriteRaw(s: string) = sb.Append(s) |> ignore
    member this.WriteStr(s: string) = this.WriteRaw(jStr s)
    member this.WriteFloat(v: float) = this.WriteRaw(jFloat v)
    member this.WriteInt(n: int) = this.WriteRaw(string n)

    member this.WriteObject(fields: (string * (unit -> unit)) list) =
        this.WriteRaw("{")
        indent <- indent + 2
        let last = fields.Length - 1
        fields |> List.iteri (fun idx (key, writeVal) ->
            this.WriteRaw("\n" + indentStr())
            this.WriteStr(key)
            this.WriteRaw(": ")
            writeVal ()
            if idx < last then this.WriteRaw(",")
        )
        indent <- indent - 2
        if not fields.IsEmpty then this.WriteRaw("\n" + indentStr())
        this.WriteRaw("}")

    member this.WriteArray(items: (unit -> unit) list) =
        this.WriteRaw("[")
        if items.IsEmpty then
            this.WriteRaw("]")
        else
            indent <- indent + 2
            let last = items.Length - 1
            items |> List.iteri (fun idx writeItem ->
                this.WriteRaw("\n" + indentStr())
                writeItem ()
                if idx < last then this.WriteRaw(",")
            )
            indent <- indent - 2
            this.WriteRaw("\n" + indentStr() + "]")

    member this.WriteStringList(ss: string list) =
        this.WriteArray(ss |> List.map (fun s -> fun () -> this.WriteStr(s)))

// ---------------------------------------------------------------------------
// Serialization of IR nodes
// ---------------------------------------------------------------------------

let rec private writeExpr (w: Writer) (expr: Expr) : unit =
    match expr with
    | Var(line, col, name) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "Var"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "name",  fun () -> w.WriteStr name
        ]
    | Const(line, col, value) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "Const"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "value", fun () -> w.WriteFloat value
        ]
    | StringLit(line, col, value) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "StringLit"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "value", fun () -> w.WriteStr value
        ]
    | Neg(line, col, operand) ->
        w.WriteObject [
            "type",    fun () -> w.WriteStr "Neg"
            "line",    fun () -> w.WriteInt line
            "col",     fun () -> w.WriteInt col
            "operand", fun () -> writeExpr w operand
        ]
    | Not(line, col, operand) ->
        w.WriteObject [
            "type",    fun () -> w.WriteStr "Not"
            "line",    fun () -> w.WriteInt line
            "col",     fun () -> w.WriteInt col
            "operand", fun () -> writeExpr w operand
        ]
    | BinOp(line, col, op, left, right) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "BinOp"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "op",    fun () -> w.WriteStr op
            "left",  fun () -> writeExpr w left
            "right", fun () -> writeExpr w right
        ]
    | Transpose(line, col, operand) ->
        w.WriteObject [
            "type",    fun () -> w.WriteStr "Transpose"
            "line",    fun () -> w.WriteInt line
            "col",     fun () -> w.WriteInt col
            "operand", fun () -> writeExpr w operand
        ]
    | FieldAccess(line, col, base_, field) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "FieldAccess"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "base",  fun () -> writeExpr w base_
            "field", fun () -> w.WriteStr field
        ]
    | Lambda(line, col, parms, body) ->
        w.WriteObject [
            "type",   fun () -> w.WriteStr "Lambda"
            "line",   fun () -> w.WriteInt line
            "col",    fun () -> w.WriteInt col
            "params", fun () -> w.WriteStringList parms
            "body",   fun () -> writeExpr w body
        ]
    | FuncHandle(line, col, name) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "FuncHandle"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "name", fun () -> w.WriteStr name
        ]
    | End(line, col) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "End"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
        ]
    | Apply(line, col, base_, args) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Apply"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "base", fun () -> writeExpr w base_
            "args", fun () -> w.WriteArray(args |> List.map (fun a -> fun () -> writeIndexArg w a))
        ]
    | CurlyApply(line, col, base_, args) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "CurlyApply"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "base", fun () -> writeExpr w base_
            "args", fun () -> w.WriteArray(args |> List.map (fun a -> fun () -> writeIndexArg w a))
        ]
    | MatrixLit(line, col, rows) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "MatrixLit"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "rows", fun () -> writeRows w rows
        ]
    | CellLit(line, col, rows) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "CellLit"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "rows", fun () -> writeRows w rows
        ]

and private writeRows (w: Writer) (rows: Expr list list) : unit =
    w.WriteArray(rows |> List.map (fun row ->
        fun () -> w.WriteArray(row |> List.map (fun e -> fun () -> writeExpr w e))
    ))

and private writeIndexArg (w: Writer) (arg: IndexArg) : unit =
    match arg with
    | Colon(line, col) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Colon"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
        ]
    | Range(line, col, start, end_) ->
        w.WriteObject [
            "type",  fun () -> w.WriteStr "Range"
            "line",  fun () -> w.WriteInt line
            "col",   fun () -> w.WriteInt col
            "start", fun () -> writeExpr w start
            "end",   fun () -> writeExpr w end_
        ]
    | IndexExpr(line, col, expr) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "IndexExpr"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "expr", fun () -> writeExpr w expr
        ]

and private writeBody (w: Writer) (stmts: Stmt list) : unit =
    w.WriteArray(stmts |> List.map (fun s -> fun () -> writeStmt w s))

and private writeStmt (w: Writer) (stmt: Stmt) : unit =
    match stmt with
    | Assign(line, col, name, expr) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Assign"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "name", fun () -> w.WriteStr name
            "expr", fun () -> writeExpr w expr
        ]
    | StructAssign(line, col, baseName, fields, expr) ->
        w.WriteObject [
            "type",      fun () -> w.WriteStr "StructAssign"
            "line",      fun () -> w.WriteInt line
            "col",       fun () -> w.WriteInt col
            "base_name", fun () -> w.WriteStr baseName
            "fields",    fun () -> w.WriteStringList fields
            "expr",      fun () -> writeExpr w expr
        ]
    | CellAssign(line, col, baseName, args, expr) ->
        w.WriteObject [
            "type",      fun () -> w.WriteStr "CellAssign"
            "line",      fun () -> w.WriteInt line
            "col",       fun () -> w.WriteInt col
            "base_name", fun () -> w.WriteStr baseName
            "args",      fun () -> w.WriteArray(args |> List.map (fun a -> fun () -> writeIndexArg w a))
            "expr",      fun () -> writeExpr w expr
        ]
    | IndexAssign(line, col, baseName, args, expr) ->
        w.WriteObject [
            "type",      fun () -> w.WriteStr "IndexAssign"
            "line",      fun () -> w.WriteInt line
            "col",       fun () -> w.WriteInt col
            "base_name", fun () -> w.WriteStr baseName
            "args",      fun () -> w.WriteArray(args |> List.map (fun a -> fun () -> writeIndexArg w a))
            "expr",      fun () -> writeExpr w expr
        ]
    | IndexStructAssign(line, col, baseName, indexArgs, indexKind, fields, expr) ->
        w.WriteObject [
            "type",        fun () -> w.WriteStr "IndexStructAssign"
            "line",        fun () -> w.WriteInt line
            "col",         fun () -> w.WriteInt col
            "base_name",   fun () -> w.WriteStr baseName
            "index_args",  fun () -> w.WriteArray(indexArgs |> List.map (fun a -> fun () -> writeIndexArg w a))
            "index_kind",  fun () -> w.WriteStr indexKind
            "fields",      fun () -> w.WriteStringList fields
            "expr",        fun () -> writeExpr w expr
        ]
    | FieldIndexAssign(line, col, baseName, prefixFields, indexArgs, indexKind, suffixFields, expr) ->
        w.WriteObject [
            "type",          fun () -> w.WriteStr "FieldIndexAssign"
            "line",          fun () -> w.WriteInt line
            "col",           fun () -> w.WriteInt col
            "base_name",     fun () -> w.WriteStr baseName
            "prefix_fields", fun () -> w.WriteStringList prefixFields
            "index_args",    fun () -> w.WriteArray(indexArgs |> List.map (fun a -> fun () -> writeIndexArg w a))
            "index_kind",    fun () -> w.WriteStr indexKind
            "suffix_fields", fun () -> w.WriteStringList suffixFields
            "expr",          fun () -> writeExpr w expr
        ]
    | ExprStmt(line, col, expr) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "ExprStmt"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "expr", fun () -> writeExpr w expr
        ]
    | If(line, col, cond, thenBody, elseBody) ->
        w.WriteObject [
            "type",      fun () -> w.WriteStr "If"
            "line",      fun () -> w.WriteInt line
            "col",       fun () -> w.WriteInt col
            "cond",      fun () -> writeExpr w cond
            "then_body", fun () -> writeBody w thenBody
            "else_body", fun () -> writeBody w elseBody
        ]
    | IfChain(line, col, conditions, bodies, elseBody) ->
        w.WriteObject [
            "type",       fun () -> w.WriteStr "IfChain"
            "line",       fun () -> w.WriteInt line
            "col",        fun () -> w.WriteInt col
            "conditions", fun () -> w.WriteArray(conditions |> List.map (fun c -> fun () -> writeExpr w c))
            "bodies",     fun () -> w.WriteArray(bodies |> List.map (fun b -> fun () -> writeBody w b))
            "else_body",  fun () -> writeBody w elseBody
        ]
    | While(line, col, cond, body) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "While"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "cond", fun () -> writeExpr w cond
            "body", fun () -> writeBody w body
        ]
    | For(line, col, var_, it, body) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "For"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
            "var",  fun () -> w.WriteStr var_
            "it",   fun () -> writeExpr w it
            "body", fun () -> writeBody w body
        ]
    | Switch(line, col, expr, cases, otherwise) ->
        let writeCases () =
            w.WriteArray(cases |> List.map (fun (cVal, cBody) ->
                fun () ->
                    w.WriteObject [
                        "value", fun () -> writeExpr w cVal
                        "body",  fun () -> writeBody w cBody
                    ]
            ))
        w.WriteObject [
            "type",      fun () -> w.WriteStr "Switch"
            "line",      fun () -> w.WriteInt line
            "col",       fun () -> w.WriteInt col
            "expr",      fun () -> writeExpr w expr
            "cases",     fun () -> writeCases ()
            "otherwise", fun () -> writeBody w otherwise
        ]
    | Try(line, col, tryBody, catchBody) ->
        w.WriteObject [
            "type",       fun () -> w.WriteStr "Try"
            "line",       fun () -> w.WriteInt line
            "col",        fun () -> w.WriteInt col
            "try_body",   fun () -> writeBody w tryBody
            "catch_body", fun () -> writeBody w catchBody
        ]
    | Break(line, col) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Break"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
        ]
    | Continue(line, col) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Continue"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
        ]
    | Return(line, col) ->
        w.WriteObject [
            "type", fun () -> w.WriteStr "Return"
            "line", fun () -> w.WriteInt line
            "col",  fun () -> w.WriteInt col
        ]
    | OpaqueStmt(line, col, targets, raw) ->
        // raw can be a string or empty list in Python; always serialize as string
        let rawStr =
            match raw with
            | s -> s
        w.WriteObject [
            "type",    fun () -> w.WriteStr "OpaqueStmt"
            "line",    fun () -> w.WriteInt line
            "col",     fun () -> w.WriteInt col
            "targets", fun () -> w.WriteStringList targets
            "raw",     fun () -> w.WriteStr rawStr
        ]
    | FunctionDef(line, col, name, parms, outputVars, body) ->
        w.WriteObject [
            "type",        fun () -> w.WriteStr "FunctionDef"
            "line",        fun () -> w.WriteInt line
            "col",         fun () -> w.WriteInt col
            "name",        fun () -> w.WriteStr name
            "params",      fun () -> w.WriteStringList parms
            "output_vars", fun () -> w.WriteStringList outputVars
            "body",        fun () -> writeBody w body
        ]
    | AssignMulti(line, col, targets, expr) ->
        w.WriteObject [
            "type",    fun () -> w.WriteStr "AssignMulti"
            "line",    fun () -> w.WriteInt line
            "col",     fun () -> w.WriteInt col
            "targets", fun () -> w.WriteStringList targets
            "expr",    fun () -> writeExpr w expr
        ]


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

let programToJson (program: Program) : string =
    let sb = StringBuilder()
    let w = Writer(sb)
    w.WriteObject [
        "type", fun () -> w.WriteStr "Program"
        "body", fun () -> writeBody w program.body
    ]
    sb.ToString()
