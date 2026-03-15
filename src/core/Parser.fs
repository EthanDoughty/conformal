module Parser

open System
open Lexer
open Ir

// Parse error exception with token span for underline-width diagnostics.
// Carries (message, startLine, startCol, endLine, endCol).
exception ParseError of string * int * int * int * int

/// A recovered parse error accumulated during error recovery.
type ParseDiagnostic = {
    message:  string
    startLine: int
    startCol:  int
    endLine:   int
    endCol:    int
}

// LHS accessor chain segment — replaces (string * obj) boxing in ParseLhsChain.
type LhsSegment =
    | Field of name: string
    | Paren of args: IndexArg list
    | Curly of args: IndexArg list

// ---------------------------------------------------------------------------
// extract_targets_from_tokens
// ---------------------------------------------------------------------------

let extractTargetsFromTokens (toks: Token list) : string list =
    if toks.IsEmpty then []
    else
        let arr = Array.ofList toks

        // Simple: ID =
        if arr.Length >= 2 && arr.[0].kind = TkId && arr.[1].value = "=" then
            [arr.[0].value]

        // Function-style: ID( ... ) =
        elif arr.Length >= 3 && arr.[0].kind = TkId && arr.[1].kind = TkLParen then
            let mutable depth = 0
            let mutable result: string list = []
            let mutable i = 0
            while i < arr.Length && result.IsEmpty do
                if arr.[i].kind = TkLParen then depth <- depth + 1
                elif arr.[i].kind = TkRParen then
                    depth <- depth - 1
                    if depth = 0 then
                        if i + 1 < arr.Length && arr.[i + 1].value = "=" then
                            result <- [arr.[0].value]
                        i <- arr.Length  // break
                i <- i + 1
            result

        // Destructuring: [ ... ] =
        elif arr.Length >= 2 && arr.[0].kind = TkLBracket then
            let mutable depth = 0
            let mutable bracketEnd = -1
            let mutable i = 0
            while i < arr.Length && bracketEnd = -1 do
                if arr.[i].kind = TkLBracket then depth <- depth + 1
                elif arr.[i].kind = TkRBracket then
                    depth <- depth - 1
                    if depth = 0 then bracketEnd <- i
                i <- i + 1

            if bracketEnd > 0 && bracketEnd + 1 < arr.Length && arr.[bracketEnd + 1].value = "=" then
                let mutable valid = true
                for j in 1 .. bracketEnd - 1 do
                    let tok = arr.[j]
                    if tok.kind <> TkId && tok.kind <> TkNewline && tok.kind <> TkComma && tok.value <> "~" && tok.kind <> TkDot then
                        valid <- false
                if valid then
                    let targets = System.Collections.Generic.List<string>()
                    let mutable j = 1
                    while j < bracketEnd do
                        let tok = arr.[j]
                        if tok.kind = TkId then
                            targets.Add(tok.value)
                            j <- j + 1
                            while j < bracketEnd && arr.[j].kind = TkDot do
                                j <- j + 1
                                if j < bracketEnd && arr.[j].kind = TkId then j <- j + 1
                        elif tok.value = "~" then
                            targets.Add("~")
                            j <- j + 1
                        else
                            j <- j + 1
                    targets |> Seq.toList
                else []
            else []
        else []


// ---------------------------------------------------------------------------
// Operator precedence table
// ---------------------------------------------------------------------------

let private precedenceTable : System.Collections.Generic.Dictionary<string,int> =
    let d = System.Collections.Generic.Dictionary<string,int>()
    d.["||"] <- 0; d.["|"] <- 1; d.["&&"] <- 2; d.["&"] <- 3
    d.["=="] <- 4; d.["~="] <- 4; d.["<"] <- 4; d.["<="] <- 4; d.[">"] <- 4; d.[">="] <- 4
    d.["+"] <- 5; d.["-"] <- 5
    d.["*"] <- 6; d.["/"] <- 6; d.[".*"] <- 6; d.["./"] <- 6; d.["\\"] <- 6
    d.[":"] <- 7
    d.["^"] <- 8; d.[".^"] <- 8
    d


// ---------------------------------------------------------------------------
// MatlabParser
// ---------------------------------------------------------------------------

type MatlabParser(tokenList: Token list, endlessFunctions: bool) =

    let tokens = Array.ofList tokenList
    let mutable pos = 0  // current token index
    let mutable lastArgSpecs: (string * int option * int option) list = []
    let recoveredErrors = ResizeArray<ParseDiagnostic>()

    /// Errors that were caught and recovered from during parsing.
    member _.RecoveredErrors = recoveredErrors |> Seq.toList

    // Token helpers
    member private _.Current() = tokens.[pos]
    member private _.AtEnd() = tokens.[pos].kind = TkEof

    member private _.Eat(expected: TokenKind) : Token =
        let tok = tokens.[pos]
        if tok.kind <> expected then
            raise (ParseError($"Expected {tokenKindName expected} at pos {tok.pos}, found {tokenKindName tok.kind} '{tok.value}'", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))
        pos <- pos + 1
        tok

    member private _.EatValue(expectedValue: string) : Token =
        let tok = tokens.[pos]
        if tok.value <> expectedValue then
            raise (ParseError($"Expected '{expectedValue}' at pos {tok.pos}, found {tokenKindName tok.kind} '{tok.value}'", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))
        pos <- pos + 1
        tok

    member private _.StartsExpr(tok: Token) : bool =
        match tok.kind with
        | TkNumber | TkId | TkString | TkEnd
        | TkLParen | TkLBracket | TkLCurly -> true
        | TkOp -> tok.value = "-" || tok.value = "+" || tok.value = "~" || tok.value = "@" || tok.value = "?"
        | _ -> false

    // Recovery: consume tokens until statement boundary
    member private this.RecoverToStmtBoundary(startLine: int, startCol: int) : Stmt =
        let consumed = System.Collections.Generic.List<Token>()
        let mutable depth = 0
        let mutable stop = false
        while not (this.AtEnd()) && not stop do
            let tok = this.Current()
            if depth = 0 then
                let isBlockEnd = tok.kind = TkEnd || tok.kind = TkElse || tok.kind = TkElseif ||
                                 tok.kind = TkCase || tok.kind = TkOtherwise || tok.kind = TkCatch
                if isBlockEnd then
                    stop <- true
                elif tok.kind = TkNewline then
                    pos <- pos + 1
                    stop <- true
                elif tok.kind = TkSemicolon then
                    pos <- pos + 1
                    stop <- true
                else
                    // Only count actual delimiters, not STRING tokens with bracket values
                    if tok.kind <> TkString then
                        if tok.kind = TkLParen || tok.kind = TkLBracket || tok.kind = TkLCurly then depth <- depth + 1
                        elif tok.kind = TkRParen || tok.kind = TkRBracket || tok.kind = TkRCurly then depth <- max 0 (depth - 1)
                    consumed.Add(tok)
                    pos <- pos + 1
            else
                if tok.kind <> TkString then
                    if tok.kind = TkLParen || tok.kind = TkLBracket || tok.kind = TkLCurly then depth <- depth + 1
                    elif tok.kind = TkRParen || tok.kind = TkRBracket || tok.kind = TkRCurly then depth <- max 0 (depth - 1)
                consumed.Add(tok)
                pos <- pos + 1
        let rawText = consumed |> Seq.map (fun t -> t.value) |> String.concat " "
        let targets = extractTargetsFromTokens (consumed |> Seq.toList)
        OpaqueStmt(loc startLine startCol, targets, rawText)

    // Classdef helpers

    member private this.SkipNewlines() =
        while not (this.AtEnd()) &&
              (this.Current().kind = TkNewline || this.Current().kind = TkSemicolon) do
            pos <- pos + 1

    member private this.SkipToEndOfLine() =
        while not (this.AtEnd()) && this.Current().kind <> TkNewline &&
              this.Current().kind <> TkEof && this.Current().kind <> TkEnd &&
              this.Current().kind <> TkSemicolon do
            pos <- pos + 1
        if not (this.AtEnd()) &&
           (this.Current().kind = TkNewline || this.Current().kind = TkSemicolon) then
            pos <- pos + 1

    // Skip optional (Attr = val, ...) attribute block after properties/methods keyword
    member private this.SkipParenAttributes() =
        if not (this.AtEnd()) && this.Current().kind = TkLParen then
            let mutable depth = 0
            let mutable finished = false
            while not (this.AtEnd()) && not finished do
                let tok = this.Current()
                pos <- pos + 1
                if tok.kind = TkLParen then depth <- depth + 1
                elif tok.kind = TkRParen then
                    depth <- depth - 1
                    if depth = 0 then finished <- true

    // Depth-count consume of a block that starts AFTER current position (we are on the keyword)
    // Consumes tokens until the matching END, then consumes the END.
    member private this.ConsumeBlock() =
        pos <- pos + 1  // skip keyword
        let blockOpeners = set [TkIf; TkFor; TkWhile; TkSwitch; TkTry; TkFunction; TkParfor]
        let idBlockOpeners = set ["methods"; "properties"; "events"; "enumeration"]
        let mutable depth = 1
        let mutable parenDepth = 0
        while not (this.AtEnd()) && depth > 0 do
            let tok = this.Current()
            pos <- pos + 1
            if tok.kind = TkLParen || tok.kind = TkLBracket || tok.kind = TkLCurly then
                parenDepth <- parenDepth + 1
            elif tok.kind = TkRParen || tok.kind = TkRBracket || tok.kind = TkRCurly then
                parenDepth <- max 0 (parenDepth - 1)
            elif parenDepth = 0 then
                if tok.kind = TkEnd then
                    depth <- depth - 1
                elif Set.contains tok.kind blockOpeners then
                    depth <- depth + 1
                elif tok.kind = TkId && Set.contains tok.value idBlockOpeners then
                    depth <- depth + 1

    member private this.ParsePropertiesBlock() : string list =
        pos <- pos + 1  // skip 'properties' keyword
        this.SkipParenAttributes()
        this.SkipNewlines()
        let props = System.Collections.Generic.List<string>()
        while not (this.AtEnd()) && this.Current().kind <> TkEnd do
            if this.Current().kind = TkId then
                props.Add(this.Current().value)
                pos <- pos + 1
                this.SkipToEndOfLine()
            else
                pos <- pos + 1
            this.SkipNewlines()
        if not (this.AtEnd()) && this.Current().kind = TkEnd then pos <- pos + 1
        props |> Seq.toList

    member private this.ParseMethodsBlock() : Stmt list =
        pos <- pos + 1  // skip 'methods' keyword
        this.SkipParenAttributes()
        this.SkipNewlines()
        let meths = System.Collections.Generic.List<Stmt>()
        while not (this.AtEnd()) && this.Current().kind <> TkEnd do
            if this.Current().kind = TkFunction then
                meths.Add(this.ParseFunctionDef())
            else
                pos <- pos + 1
            this.SkipNewlines()
        if not (this.AtEnd()) && this.Current().kind = TkEnd then pos <- pos + 1
        meths |> Seq.toList

    member private this.ParseClassdef() : Stmt list =
        let startTok = this.Current()
        let line = startTok.line
        let col = startTok.col
        try
            pos <- pos + 1  // skip 'classdef'
            // Skip optional (Sealed), (Abstract), etc. attribute block before class name
            this.SkipParenAttributes()
            // Class name
            let className =
                if not (this.AtEnd()) && this.Current().kind = TkId then
                    let n = this.Current().value
                    pos <- pos + 1
                    n
                else ""
            // Optional superclass: classdef Foo < Bar
            let mutable superName = ""
            if not (this.AtEnd()) && this.Current().value = "<" then
                pos <- pos + 1  // skip '<'
                if not (this.AtEnd()) && this.Current().kind = TkId then
                    superName <- this.Current().value
                    pos <- pos + 1
                    while not (this.AtEnd()) && this.Current().kind = TkDot do
                        pos <- pos + 1  // skip '.'
                        if not (this.AtEnd()) && this.Current().kind = TkId then
                            superName <- $"{superName}.{this.Current().value}"
                            pos <- pos + 1
            this.SkipNewlines()
            let mutable properties : string list = []
            let mutable methodDefs : Stmt list = []
            while not (this.AtEnd()) && this.Current().kind <> TkEnd do
                let cur = this.Current()
                if cur.kind = TkId && cur.value = "properties" then
                    properties <- properties @ this.ParsePropertiesBlock()
                elif cur.kind = TkId && cur.value = "methods" then
                    methodDefs <- methodDefs @ this.ParseMethodsBlock()
                elif cur.kind = TkId && (cur.value = "events" || cur.value = "enumeration") then
                    this.ConsumeBlock()
                else
                    pos <- pos + 1
                this.SkipNewlines()
            if not (this.AtEnd()) && this.Current().kind = TkEnd then pos <- pos + 1
            this.SkipNewlines()
            // Encode metadata in OpaqueStmt raw string
            let propPart = properties |> String.concat ","
            let superPart = if superName <> "" then $":{superName}" else ""
            let raw = $"classdef:{className}:{propPart}{superPart}"
            let opaque = OpaqueStmt(loc line col, [], raw)
            opaque :: methodDefs
        with
        | ex ->
            // Accumulate the error if it's a ParseError
            match ex with
            | :? ParseError as pe ->
                let (msg, sl, sc, el, ec) = (pe.Data0, pe.Data1, pe.Data2, pe.Data3, pe.Data4)
                recoveredErrors.Add({ message = msg; startLine = sl; startCol = sc; endLine = el; endCol = ec })
            | _ -> ()
            // Fallback: depth-count consume the rest as opaque
            let blockOpeners = set [TkIf; TkFor; TkWhile; TkSwitch; TkTry; TkFunction; TkParfor]
            let idBlockOpeners = set ["methods"; "properties"; "events"; "enumeration"]
            let mutable depth = 1
            let mutable parenDepth = 0
            let mutable finished = false
            while not (this.AtEnd()) && not finished do
                let tok = this.Current()
                pos <- pos + 1
                if tok.kind = TkLParen || tok.kind = TkLBracket || tok.kind = TkLCurly then
                    parenDepth <- parenDepth + 1
                elif tok.kind = TkRParen || tok.kind = TkRBracket || tok.kind = TkRCurly then
                    parenDepth <- max 0 (parenDepth - 1)
                elif parenDepth = 0 then
                    if tok.kind = TkEnd then
                        depth <- depth - 1
                        if depth = 0 then finished <- true
                    elif Set.contains tok.kind blockOpeners then
                        depth <- depth + 1
                    elif tok.kind = TkId && Set.contains tok.value idBlockOpeners then
                        depth <- depth + 1
            this.SkipNewlines()
            [OpaqueStmt(loc line col, [], "classdef")]

    // -------------------------------------------------------------------
    // Top-level program
    // -------------------------------------------------------------------

    member this.ParseProgram() : Program =
        let items = System.Collections.Generic.List<Stmt>()
        while not (this.AtEnd()) do
            // Skip blank lines / semicolons
            let mutable skipping = true
            while skipping && not (this.AtEnd()) do
                let cur = this.Current()
                if cur.kind = TkNewline then pos <- pos + 1
                elif cur.kind = TkSemicolon then pos <- pos + 1
                else skipping <- false
            if not (this.AtEnd()) then
                let cur = this.Current()
                if cur.kind = TkId && cur.value = "classdef" then
                    for stmt in this.ParseClassdef() do items.Add(stmt)
                elif cur.kind = TkFunction then
                    items.Add(this.ParseFunctionDef())
                else
                    let savedPos = pos
                    items.Add(this.ParseStmt())
                    if pos = savedPos then pos <- pos + 1
        { body = items |> Seq.toList }

    // -------------------------------------------------------------------
    // Function definition
    // -------------------------------------------------------------------

    // Eat a function name: accepts ID or END (MATLAB allows `end` as a function name
    // for operator overloading, e.g. @classname/end.m).
    member private this.EatFuncName() : string =
        let tok = this.Current()
        if tok.kind = TkId || tok.kind = TkEnd then
            pos <- pos + 1
            tok.value
        else
            raise (ParseError($"Expected function name at pos {tok.pos}, found {tokenKindName tok.kind} '{tok.value}'", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))

    member private this.ParseFunctionDef() : Stmt =
        let funcTok = this.Eat(TkFunction)
        let line = funcTok.line
        let col = funcTok.col
        let outputVars = ResizeArray<string>()
        let mutable name = ""
        let mutable hasParens = true

        if this.Current().kind = TkId || this.Current().kind = TkEnd then
            let lookahead = if pos + 1 < tokens.Length then tokens.[pos + 1] else tokens.[tokens.Length - 1]
            if lookahead.kind = TkLParen then
                // Procedure form: function name(args) — or function end(args)
                name <- this.EatFuncName()
                this.Eat(TkLParen) |> ignore
            elif lookahead.value = "=" && this.Current().kind = TkId then
                // Single return: function result = name(args)
                // Note: only ID can be an output var (not END)
                outputVars.Add(this.Eat(TkId).value)
                this.EatValue("=") |> ignore
                name <- this.EatFuncName()
                if this.Current().kind = TkLParen then this.Eat(TkLParen) |> ignore
                else hasParens <- false
            elif lookahead.kind = TkNewline || lookahead.kind = TkEof || lookahead.kind = TkSemicolon ||
                 lookahead.kind = TkFunction then
                // No-arg procedure
                name <- this.EatFuncName()
                hasParens <- false
            else
                let errTok = this.Current()
                raise (ParseError("Expected '=' or '(' after function name at " + string errTok.pos, errTok.line, errTok.col, errTok.line, errTok.col + max 1 errTok.value.Length))
        elif this.Current().kind = TkLBracket then
            // Multiple outputs: function [a, b] = name(args)
            this.Eat(TkLBracket) |> ignore
            if this.Current().kind = TkRBracket then
                this.Eat(TkRBracket) |> ignore
            else
                outputVars.Add(this.Eat(TkId).value)
                while this.Current().kind = TkComma || this.Current().kind = TkId do
                    if this.Current().kind = TkComma then this.Eat(TkComma) |> ignore
                    outputVars.Add(this.Eat(TkId).value)
                this.Eat(TkRBracket) |> ignore
            this.EatValue("=") |> ignore
            name <- this.EatFuncName()
            if this.Current().kind = TkLParen then this.Eat(TkLParen) |> ignore
            else hasParens <- false
        else
            let errTok = this.Current()
            raise (ParseError("Expected function output or name at " + string errTok.pos, errTok.line, errTok.col, errTok.line, errTok.col + max 1 errTok.value.Length))

        // Parameters
        let parms = ResizeArray<string>()
        if hasParens then
            let eatParam () =
                if this.Current().value = "~" then
                    this.EatValue("~") |> ignore; "~"
                else this.Eat(TkId).value
            if this.Current().kind <> TkRParen then
                parms.Add(eatParam ())
                while this.Current().kind = TkComma do
                    this.Eat(TkComma) |> ignore
                    parms.Add(eatParam ())
            this.Eat(TkRParen) |> ignore

        // Skip newline/semicolon after signature
        if this.Current().kind = TkNewline then this.Eat(TkNewline) |> ignore
        elif this.Current().kind = TkSemicolon then pos <- pos + 1

        // Body — reset lastArgSpecs so ParseArgumentsBlock can populate it
        lastArgSpecs <- []
        let body =
            if endlessFunctions then this.ParseBlock([| TkFunction |])
            else
                let b = this.ParseBlock([| TkEnd |])
                this.Eat(TkEnd) |> ignore
                b
        let argAnns = lastArgSpecs
        lastArgSpecs <- []

        FunctionDef(loc line col, name, Seq.toList parms, Seq.toList outputVars, body, argAnns)

    // -------------------------------------------------------------------
    // Statements
    // -------------------------------------------------------------------

    member this.ParseStmt() : Stmt =
        let tok = this.Current()
        let startLine = tok.line
        let startCol = tok.col
        let startPos = pos
        try
            match tok.kind with
            | TkFor    -> this.ParseFor(false)
            | TkParfor -> this.ParseFor(true)
            | TkGlobal | TkPersistent -> this.ParseGlobal()
            | TkWhile  -> this.ParseWhile()
            | TkIf     -> this.ParseIf()
            | TkSwitch -> this.ParseSwitch()
            | TkTry    -> this.ParseTry()
            | TkBreak  -> let t = this.Eat(TkBreak) in Break(loc t.line t.col)
            | TkContinue -> let t = this.Eat(TkContinue) in Continue(loc t.line t.col)
            | TkReturn -> let t = this.Eat(TkReturn) in Return(loc t.line t.col)
            | TkFunction -> this.ParseFunctionDef()
            | TkShellEscape ->
                // !command — skip the token and any trailing newline
                let t = tokens.[pos]
                pos <- pos + 1
                if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
                ExprStmt(loc t.line t.col, Const(loc t.line t.col, 0.0))
            | TkNewline ->
                this.Eat(TkNewline) |> ignore
                ExprStmt(loc 0 0, Const(loc 0 0, 0.0))
            | TkId when tok.value = "arguments" &&
                        (let nk = tokens.[pos + 1].kind
                         nk = TkNewline ||
                         (nk = TkLParen && pos + 2 < tokens.Length &&
                          tokens.[pos + 2].kind = TkId &&
                          (let q = tokens.[pos + 2].value
                           q = "Input" || q = "Output" || q = "Repeating"))) ->
                this.ParseArgumentsBlock()
            | _ ->
                let node = this.ParseSimpleStmt()
                let curKind = this.Current().kind
                if curKind <> TkNewline && curKind <> TkEof && curKind <> TkSemicolon then
                    pos <- startPos
                    this.RecoverToStmtBoundary(startLine, startCol)
                else
                    if this.Current().kind = TkNewline then this.Eat(TkNewline) |> ignore
                    elif this.Current().kind = TkSemicolon then pos <- pos + 1
                    node
        with
        | :? ParseError as pe ->
            let (msg, sl, sc, el, ec) = (pe.Data0, pe.Data1, pe.Data2, pe.Data3, pe.Data4)
            recoveredErrors.Add({ message = msg; startLine = sl; startCol = sc; endLine = el; endCol = ec })
            pos <- startPos
            this.RecoverToStmtBoundary(startLine, startCol)

    // Bracket stmt: [a,b] = expr  or  matrix expression
    member private this.ParseBracketStmt() : Stmt =
        let savedPos = pos
        try
            this.Eat(TkLBracket) |> ignore
            let eatTarget () =
                if this.Current().value = "~" then
                    this.EatValue("~") |> ignore; "~"
                else
                    let mutable tname = this.Eat(TkId).value
                    while this.Current().kind = TkDot do
                        this.Eat(TkDot) |> ignore
                        let nextId = this.Eat(TkId).value
                        tname <- $"{tname}.{nextId}"
                    tname
            let targets = ResizeArray<string>()
            targets.Add(eatTarget ())
            while this.Current().kind = TkComma || this.Current().kind = TkId || this.Current().value = "~" do
                if this.Current().kind = TkComma then this.Eat(TkComma) |> ignore
                targets.Add(eatTarget ())
            this.Eat(TkRBracket) |> ignore
            if this.Current().value = "=" then
                let eqTok = this.EatValue("=")
                let expr = this.ParseExpr(0, false, true)
                AssignMulti(loc eqTok.line eqTok.col, Seq.toList targets, expr)
            else
                pos <- savedPos
                let expr = this.ParseExpr(0, false, true)
                ExprStmt(loc expr.Line expr.Col, expr)
        with
        | :? ParseError ->
            pos <- savedPos
            let expr = this.ParseExpr(0, false, true)
            ExprStmt(loc expr.Line expr.Col, expr)

    // Greedy LHS accessor chain after an ID.
    member private this.ParseLhsChain() : LhsSegment list =
        let chain = System.Collections.Generic.List<LhsSegment>()
        let mutable stop = false
        while not stop do
            let cur = this.Current()
            if cur.kind = TkDot then
                this.Eat(TkDot) |> ignore
                if this.Current().kind = TkId then
                    chain.Add(Field(this.Eat(TkId).value))
                elif this.Current().kind = TkLParen then
                    this.Eat(TkLParen) |> ignore
                    this.ParseExpr(0, false, true) |> ignore
                    this.Eat(TkRParen) |> ignore
                    chain.Add(Field("<dynamic>"))
                else
                    stop <- true
            elif cur.kind = TkLParen then
                this.Eat(TkLParen) |> ignore
                let args = this.ParseParenArgs()
                this.Eat(TkRParen) |> ignore
                chain.Add(Paren(args))
            elif cur.kind = TkLCurly then
                this.Eat(TkLCurly) |> ignore
                let args = this.ParseParenArgs()
                this.Eat(TkRCurly) |> ignore
                chain.Add(Curly(args))
            else
                stop <- true
        chain |> Seq.toList

    member private _.ChainToExpr(idTok: Token, chain: LhsSegment list) : Expr =
        let mutable expr: Expr = Var(loc idTok.line idTok.col, idTok.value)
        for seg in chain do
            match seg with
            | Field name -> expr <- FieldAccess(loc idTok.line idTok.col, expr, name)
            | Paren args -> expr <- Apply(loc idTok.line idTok.col, expr, args)
            | Curly args -> expr <- CurlyApply(loc idTok.line idTok.col, expr, args)
        expr

    member private _.ClassifyAssignment(idTok: Token, chain: LhsSegment list, eqTok: Token, rhs: Expr) : Stmt =
        let baseName = idTok.value
        let isField = function Field _ -> true | _ -> false
        let fieldName = function Field n -> n | _ -> ""
        let indexArgs = function Paren a | Curly a -> a | _ -> []
        let indexKind = function Paren _ -> "paren" | Curly _ -> "curly" | _ -> ""

        if chain.IsEmpty then
            Assign(loc idTok.line idTok.col, baseName, rhs)

        elif chain |> List.forall isField then
            StructAssign(loc eqTok.line eqTok.col, baseName, chain |> List.map fieldName, rhs)

        elif chain.Length = 1 then
            match chain.[0] with
            | Paren args -> IndexAssign(loc eqTok.line eqTok.col, baseName, args, rhs)
            | Curly args -> CellAssign(loc eqTok.line eqTok.col, baseName, args, rhs)
            | Field _ -> StructAssign(loc eqTok.line eqTok.col, baseName, [fieldName chain.[0]], rhs)

        elif (match chain.[0] with Paren _ | Curly _ -> true | _ -> false) && chain.Length > 1 &&
             chain.[1..] |> List.forall isField then
            IndexStructAssign(loc eqTok.line eqTok.col, baseName, indexArgs chain.[0],
                              indexKind chain.[0], chain.[1..] |> List.map fieldName, rhs)

        else
            let indexPos = chain |> List.tryFindIndex (fun s -> match s with Paren _ | Curly _ -> true | _ -> false)
            match indexPos with
            | Some ip ->
                let prefix = if ip > 0 then chain.[..ip-1] else []
                let suffix = if ip < chain.Length - 1 then chain.[ip+1..] else []
                if not prefix.IsEmpty && prefix |> List.forall isField &&
                   not suffix.IsEmpty && suffix |> List.forall isField then
                    FieldIndexAssign(
                        loc eqTok.line eqTok.col, baseName,
                        prefix |> List.map fieldName,
                        indexArgs chain.[ip],
                        indexKind chain.[ip],
                        suffix |> List.map fieldName,
                        rhs)
                elif chain.Length >= 2 && (match chain.[chain.Length-1] with Paren _ | Curly _ -> true | _ -> false) &&
                     chain.[..chain.Length-2] |> List.forall isField then
                    ExprStmt(loc eqTok.line eqTok.col, rhs)
                else
                    OpaqueStmt(loc eqTok.line eqTok.col, [baseName], "")
            | None ->
                OpaqueStmt(loc eqTok.line eqTok.col, [baseName], "")

    member private this.ParseSimpleStmt() : Stmt =
        if this.Current().kind = TkLBracket then
            this.ParseBracketStmt()
        elif this.Current().kind = TkId then
            let idTok = this.Eat(TkId)
            let chain = this.ParseLhsChain()
            if this.Current().value = "=" then
                let eqTok = this.EatValue("=")
                let rhs = this.ParseExpr(0, false, true)
                this.ClassifyAssignment(idTok, chain, eqTok, rhs)
            else
                let mutable expr = this.ChainToExpr(idTok, chain)
                expr <- this.ParsePostfix(expr)
                expr <- this.ParseExprRest(expr, 0, false, true)
                ExprStmt(loc expr.Line expr.Col, expr)
        else
            let expr = this.ParseExpr(0, false, true)
            ExprStmt(loc expr.Line expr.Col, expr)

    // -------------------------------------------------------------------
    // arguments block (R2019b+)
    // -------------------------------------------------------------------

    /// Parse an `arguments ... end` block, extracting parameter size annotations.
    /// For (Input) or plain blocks, extracts (paramName, rowDim, colDim) tuples from
    /// size annotations like `x (1,:) double`. For (Output) or (Repeating) blocks,
    /// skips without extracting. The extracted specs are stored in lastArgSpecs.
    member private this.ParseArgumentsBlock() : Stmt =
        let argTok = tokens.[pos]
        pos <- pos + 1 // consume 'arguments'
        // Check optional qualifier: (Input), (Output), (Repeating)
        let mutable isOutput = false
        if this.Current().kind = TkLParen then
            pos <- pos + 1 // consume '('
            if this.Current().kind = TkId then
                let q = this.Current().value
                if q = "Output" || q = "Repeating" then isOutput <- true
            while this.Current().kind <> TkRParen && not (this.AtEnd()) do
                pos <- pos + 1
            if this.Current().kind = TkRParen then pos <- pos + 1
        // Skip leading newline after 'arguments' or 'arguments(...)'
        if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
        if isOutput then
            // For Output/Repeating blocks, just skip to 'end'
            let mutable depth = 1
            while depth > 0 && not (this.AtEnd()) do
                let tk = this.Current()
                if tk.kind = TkEnd then depth <- depth - 1
                if depth > 0 then pos <- pos + 1
            if this.Current().kind = TkEnd then pos <- pos + 1
            if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
        else
            // Parse parameter lines until 'end'
            let specs = ResizeArray<string * int option * int option>()
            while not (this.AtEnd()) && this.Current().kind <> TkEnd do
                // Skip blank lines
                if this.Current().kind = TkNewline then
                    pos <- pos + 1
                elif this.Current().kind = TkId then
                    let paramName = this.Current().value
                    pos <- pos + 1
                    // Check for name-value pair: opts.Color — skip entire line
                    if not (this.AtEnd()) && this.Current().kind = TkDot then
                        // Skip rest of line
                        while not (this.AtEnd()) && this.Current().kind <> TkNewline && this.Current().kind <> TkEnd do
                            pos <- pos + 1
                        if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
                    else
                        // Check for size tuple: (rows, cols)
                        if not (this.AtEnd()) && this.Current().kind = TkLParen then
                            pos <- pos + 1 // consume '('
                            // Parse first dimension: number or ':'
                            let mutable dim1 : int option = None
                            if this.Current().kind = TkNumber then
                                dim1 <- Some (int (System.Double.Parse(this.Current().value)))
                                pos <- pos + 1
                            elif this.Current().kind = TkOp && this.Current().value = ":" then
                                pos <- pos + 1
                            // Consume comma
                            if not (this.AtEnd()) && this.Current().kind = TkComma then pos <- pos + 1
                            // Parse second dimension: number or ':'
                            let mutable dim2 : int option = None
                            if this.Current().kind = TkNumber then
                                dim2 <- Some (int (System.Double.Parse(this.Current().value)))
                                pos <- pos + 1
                            elif this.Current().kind = TkOp && this.Current().value = ":" then
                                pos <- pos + 1
                            // Consume closing ')'
                            if not (this.AtEnd()) && this.Current().kind = TkRParen then pos <- pos + 1
                            specs.Add(paramName, dim1, dim2)
                        // Skip rest of line (type name, validators, defaults)
                        while not (this.AtEnd()) && this.Current().kind <> TkNewline && this.Current().kind <> TkEnd do
                            pos <- pos + 1
                        if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
                else
                    // Skip unexpected tokens (safety)
                    pos <- pos + 1
            if this.Current().kind = TkEnd then pos <- pos + 1
            if not (this.AtEnd()) && this.Current().kind = TkNewline then pos <- pos + 1
            lastArgSpecs <- Seq.toList specs
        ExprStmt(loc argTok.line argTok.col, Const(loc argTok.line argTok.col, 0.0))

    // -------------------------------------------------------------------
    // Control flow
    // -------------------------------------------------------------------

    member private this.ParseFor(isParfor: bool) : Stmt =
        this.Eat(if isParfor then TkParfor else TkFor) |> ignore
        // MATLAB allows optional parens: for(i = 1:n) ... end
        let hasParen = this.Current().kind = TkLParen
        if hasParen then pos <- pos + 1
        let varTok = this.Eat(TkId)
        this.EatValue("=") |> ignore
        let it = this.ParseExpr(0, false, true)
        if hasParen then this.Eat(TkRParen) |> ignore
        let body = this.ParseBlock([| TkEnd |])
        this.Eat(TkEnd) |> ignore
        For(loc it.Line it.Col, varTok.value, it, body)

    member private this.ParseGlobal() : Stmt =
        let kwTok = tokens.[pos]
        pos <- pos + 1
        let varNames = ResizeArray<string>()
        while this.Current().kind = TkId do
            varNames.Add(this.Eat(TkId).value)
        let varNamesList = Seq.toList varNames
        let rawText = kwTok.value + " " + String.concat " " varNamesList
        // Consume trailing semicolon or newline so the caller doesn't see a stray token.
        if not (this.AtEnd()) then
            let cur = this.Current()
            if cur.kind = TkNewline then pos <- pos + 1
            elif cur.kind = TkSemicolon then pos <- pos + 1
        OpaqueStmt(loc kwTok.line kwTok.col, varNamesList, rawText)

    member private this.ParseWhile() : Stmt =
        this.Eat(TkWhile) |> ignore
        let cond = this.ParseExpr(0, false, true)
        let body = this.ParseBlock([| TkEnd |])
        this.Eat(TkEnd) |> ignore
        While(loc cond.Line cond.Col, cond, body)

    member private this.ParseIf() : Stmt =
        this.Eat(TkIf) |> ignore
        let cond = this.ParseExpr(0, false, true)
        let thenBody = this.ParseBlock([| TkElse; TkElseif; TkEnd |])
        let elseifs = ResizeArray<Expr * Stmt list>()
        while this.Current().kind = TkElseif do
            this.Eat(TkElseif) |> ignore
            let elifCond = this.ParseExpr(0, false, true)
            let elifBody = this.ParseBlock([| TkElse; TkElseif; TkEnd |])
            elseifs.Add((elifCond, elifBody))
        let skipStmt = ExprStmt(loc 0 0, Const(loc 0 0, 0.0))
        let mutable elseBody = [skipStmt]
        if this.Current().kind = TkElse then
            this.Eat(TkElse) |> ignore
            elseBody <- this.ParseBlock([| TkEnd |])
        this.Eat(TkEnd) |> ignore
        if elseifs.Count = 0 then
            If(loc cond.Line cond.Col, cond, thenBody, elseBody)
        else
            let elseifsList = Seq.toList elseifs
            let conditions = cond :: (elseifsList |> List.map fst)
            let bodies = thenBody :: (elseifsList |> List.map snd)
            IfChain(loc cond.Line cond.Col, conditions, bodies, elseBody)

    member private this.ParseSwitch() : Stmt =
        this.Eat(TkSwitch) |> ignore
        let expr = this.ParseExpr(0, false, true)
        // Skip all newlines (blank lines, comment-only lines stripped by lexer)
        while this.Current().kind = TkNewline do pos <- pos + 1
        let cases = ResizeArray<Expr * Stmt list>()
        while this.Current().kind = TkCase do
            this.Eat(TkCase) |> ignore
            let caseVal = this.ParseExpr(0, false, true)
            let caseBody = this.ParseBlock([| TkCase; TkOtherwise; TkEnd |])
            cases.Add((caseVal, caseBody))
        let skipStmt = ExprStmt(loc 0 0, Const(loc 0 0, 0.0))
        let mutable otherwiseBody = [skipStmt]
        if this.Current().kind = TkOtherwise then
            this.Eat(TkOtherwise) |> ignore
            otherwiseBody <- this.ParseBlock([| TkEnd |])
        this.Eat(TkEnd) |> ignore
        Switch(loc expr.Line expr.Col, expr, Seq.toList cases, otherwiseBody)

    member private this.ParseTry() : Stmt =
        this.Eat(TkTry) |> ignore
        let tryBody = this.ParseBlock([| TkCatch; TkEnd |])
        let skipStmt = ExprStmt(loc 0 0, Const(loc 0 0, 0.0))
        let mutable catchBody = [skipStmt]
        if this.Current().kind = TkCatch then
            this.Eat(TkCatch) |> ignore
            if this.Current().kind = TkId then this.Eat(TkId) |> ignore
            catchBody <- this.ParseBlock([| TkEnd |])
        this.Eat(TkEnd) |> ignore
        let l, c =
            match tryBody with
            | h :: _ -> h.Line, h.Col
            | [] -> 0, 0
        Try(loc l c, tryBody, catchBody)

    member private this.ParseBlock(untilKinds: TokenKind[]) : Stmt list =
        if this.Current().kind = TkNewline then this.Eat(TkNewline) |> ignore
        let stmts = System.Collections.Generic.List<Stmt>()
        while not (this.AtEnd()) && not (Array.contains (this.Current().kind) untilKinds) do
            let savedPos = pos
            stmts.Add(this.ParseStmt())
            if pos = savedPos then pos <- pos + 1
        if stmts.Count = 0 then
            [ExprStmt(loc 0 0, Const(loc 0 0, 0.0))]
        else
            stmts |> Seq.toList

    // -------------------------------------------------------------------
    // Expressions
    // -------------------------------------------------------------------

    member private this.ParseExpr(minPrec: int, matrixContext: bool, colonVisible: bool) : Expr =
        let left = this.ParsePrefix(matrixContext, colonVisible)
        this.ParseExprRest(left, minPrec, matrixContext, colonVisible)

    member private this.ParsePrefix(matrixContext: bool, colonVisible: bool) : Expr =
        let tok = this.Current()
        // Check kind-based literal types first to prevent STRING/NUMBER values
        // from matching operator arms (e.g. STRING "(" matching the "(" case).
        match tok.kind with
        | TkNumber ->
            let numTok = tokens.[pos]
            pos <- pos + 1
            Const(loc numTok.line numTok.col, float numTok.value)

        | TkString ->
            let strTok = tokens.[pos]
            pos <- pos + 1
            StringLit(loc strTok.line strTok.col, strTok.value)

        | TkId ->
            let idTok = tokens.[pos]
            pos <- pos + 1
            let left: Expr = Var(loc idTok.line idTok.col, idTok.value)
            this.ParsePostfix(left)

        | TkEnd ->
            let endTok = tokens.[pos]
            pos <- pos + 1
            End(loc endTok.line endTok.col)

        | TkLParen ->
            this.Eat(TkLParen) |> ignore
            let inner = this.ParseExpr(0, false, true)
            this.Eat(TkRParen) |> ignore
            this.ParsePostfix(inner)

        | TkLBracket ->
            let ml = this.ParseMatrixLiteral()
            this.ParsePostfix(ml)

        | TkLCurly ->
            let cl = this.ParseCellLiteral()
            this.ParsePostfix(cl)

        | TkOp ->
            // Value-based matching for operators (safe now: STRING/NUMBER/ID/END/delimiters handled above)
            match tok.value with
            | "+" ->
                pos <- pos + 1
                this.ParseExpr(precedenceTable.["+"], matrixContext, colonVisible)

            | "-" ->
                let minusTok = tokens.[pos]
                pos <- pos + 1
                let operand = this.ParseExpr(precedenceTable.["-"], matrixContext, colonVisible)
                Neg(loc minusTok.line minusTok.col, operand)

            | "~" ->
                let notTok = tokens.[pos]
                pos <- pos + 1
                let operand = this.ParseExpr(precedenceTable.["+"], matrixContext, colonVisible)
                Not(loc notTok.line notTok.col, operand)

            | "@" ->
                let atTok = tokens.[pos]
                pos <- pos + 1
                let next = this.Current()
                if next.kind = TkLParen then
                    this.Eat(TkLParen) |> ignore
                    let parms = ResizeArray<string>()
                    if this.Current().kind <> TkRParen then
                        parms.Add(this.Eat(TkId).value)
                        while this.Current().kind = TkComma do
                            this.Eat(TkComma) |> ignore
                            parms.Add(this.Eat(TkId).value)
                    this.Eat(TkRParen) |> ignore
                    let body = this.ParseExpr(0, false, true)
                    Lambda(loc atTok.line atTok.col, Seq.toList parms, body)
                elif next.kind = TkId then
                    let name = this.Eat(TkId).value
                    FuncHandle(loc atTok.line atTok.col, name)
                else
                    raise (ParseError("Expected '(' or function name after '@' at " + string next.pos, next.line, next.col, next.line, next.col + max 1 next.value.Length))

            | "?" ->
                let qTok = tokens.[pos]
                pos <- pos + 1
                if this.Current().kind = TkId then
                    let name = this.Eat(TkId).value
                    MetaClass(loc qTok.line qTok.col, name)
                else
                    Var(loc qTok.line qTok.col, "?")

            | _ ->
                raise (ParseError($"Unexpected token {tokenKindName tok.kind} '{tok.value}' in expression at {tok.pos}", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))

        | _ ->
            raise (ParseError($"Unexpected token {tokenKindName tok.kind} '{tok.value}' in expression at {tok.pos}", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))

    member private this.ParsePostfix(initial: Expr) : Expr =
        let mutable left = initial
        let mutable stop = false
        while not stop do
            let tok = this.Current()
            if tok.kind = TkLParen then
                let lparenTok = tokens.[pos]
                pos <- pos + 1
                let args = this.ParseParenArgs()
                this.Eat(TkRParen) |> ignore
                left <- Apply(loc lparenTok.line lparenTok.col, left, args)

            elif tok.kind = TkLCurly then
                let lcurlyTok = tokens.[pos]
                pos <- pos + 1
                let args = this.ParseParenArgs()
                this.Eat(TkRCurly) |> ignore
                left <- CurlyApply(loc lcurlyTok.line lcurlyTok.col, left, args)

            elif tok.kind = TkTranspose then
                let tTok = tokens.[pos]
                pos <- pos + 1
                left <- Transpose(loc tTok.line tTok.col, left)

            elif tok.kind = TkDotOp && tok.value = ".'" then
                let tTok = tokens.[pos]
                pos <- pos + 1
                left <- Transpose(loc tTok.line tTok.col, left)

            elif tok.kind = TkDot then
                let dotTok = tokens.[pos]
                pos <- pos + 1
                if this.Current().kind = TkId then
                    let fieldName = this.Eat(TkId).value
                    left <- FieldAccess(loc dotTok.line dotTok.col, left, fieldName)
                elif this.Current().kind = TkLParen then
                    this.Eat(TkLParen) |> ignore
                    this.ParseExpr(0, false, true) |> ignore
                    this.Eat(TkRParen) |> ignore
                    left <- FieldAccess(loc dotTok.line dotTok.col, left, "<dynamic>")
                else
                    raise (ParseError("Expected field name after '.' at " + string tok.pos, tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))

            else
                stop <- true
        left

    member private this.ParseExprRest(initial: Expr, minPrec: int, matrixContext: bool, colonVisible: bool) : Expr =
        let mutable left = initial
        let mutable stop = false
        while not stop do
            let tok = this.Current()
            let op = tok.value
            let mutable prec = 0
            // Guard: only treat as operator if the token IS an operator.
            // TkOp tokens have kind = TkOp.
            // TkDotOp tokens have kind = TkDotOp (e.g., kind=TkDotOp, value=".*").
            // STRING/NUMBER/ID tokens must not be confused with operators even if
            // their value matches (e.g., STRING '*' vs OP '*').
            let isOp = match tok.kind with TkOp | TkDotOp -> true | _ -> false
            if not isOp || not (precedenceTable.TryGetValue(op, &prec)) then
                stop <- true
            elif op = ":" && not colonVisible then
                stop <- true
            elif prec < minPrec then
                stop <- true
            else
                // Matrix context: space before +/- but no space after => new element
                let breakForMatrix =
                    if matrixContext && (op = "+" || op = "-") && pos >= 1 then
                        let prevTok = tokens.[pos - 1]
                        let prevEnd = prevTok.pos + prevTok.value.Length
                        let opStart = tok.pos
                        let spaceBefore = opStart > prevEnd
                        if spaceBefore && pos + 1 < tokens.Length then
                            let nextTok = tokens.[pos + 1]
                            let opEnd = opStart + op.Length
                            not (nextTok.pos > opEnd)  // break if no space after
                        else false
                    else false
                if breakForMatrix then
                    stop <- true
                else
                    let opTok = tokens.[pos]
                    pos <- pos + 1
                    let rightPrec = if op = "^" || op = ".^" then prec else prec + 1
                    let right = this.ParseExpr(rightPrec, false, colonVisible)
                    left <- BinOp(loc opTok.line opTok.col, op, left, right)
        left

    // -------------------------------------------------------------------
    // Matrix / cell literals
    // -------------------------------------------------------------------

    member private this.ParseDelimitedRows(endKind: TokenKind) : int * int * Expr list list =
        let cur = this.Current()
        let line = cur.line
        let col = cur.col

        if this.Current().kind = endKind then
            (line, col, [])
        else
            let rows = System.Collections.Generic.List<Expr list>()
            let mutable outerStop = false
            while not outerStop do
                let row = System.Collections.Generic.List<Expr>()
                row.Add(this.ParseExpr(0, true, true))
                let mutable innerStop = false
                while not innerStop do
                    let tok = this.Current()
                    if tok.kind = TkComma then
                        pos <- pos + 1
                        row.Add(this.ParseExpr(0, true, true))
                    elif tok.kind = TkSemicolon || tok.kind = endKind || tok.kind = TkNewline || tok.kind = TkEof then
                        innerStop <- true
                    elif this.StartsExpr(tok) then
                        row.Add(this.ParseExpr(0, true, true))
                    else
                        innerStop <- true
                rows.Add(row |> Seq.toList)

                if this.Current().kind = endKind then
                    outerStop <- true
                elif this.Current().kind = TkSemicolon then
                    pos <- pos + 1
                    // Skip newlines after semicolon row separator (multiline matrix literals)
                    while this.Current().kind = TkNewline do pos <- pos + 1
                    if this.Current().kind = endKind then outerStop <- true
                elif this.Current().kind = TkNewline then
                    pos <- pos + 1
                    if this.Current().kind = endKind then outerStop <- true
                else
                    let tok = this.Current()
                    raise (ParseError($"Unexpected token {tokenKindName tok.kind} '{tok.value}' in literal at {tok.pos}", tok.line, tok.col, tok.line, tok.col + max 1 tok.value.Length))
            (line, col, rows |> Seq.toList)

    member private this.ParseMatrixLiteral() : Expr =
        let lbrack = this.Eat(TkLBracket)
        let line, col, rows = this.ParseDelimitedRows(TkRBracket)
        this.Eat(TkRBracket) |> ignore
        let l, c = if rows.IsEmpty then lbrack.line, lbrack.col else line, col
        MatrixLit(loc l c, rows)

    member private this.ParseCellLiteral() : Expr =
        let lcurly = this.Eat(TkLCurly)
        let line, col, rows = this.ParseDelimitedRows(TkRCurly)
        this.Eat(TkRCurly) |> ignore
        let l, c = if rows.IsEmpty then lcurly.line, lcurly.col else line, col
        CellLit(loc l c, rows)

    // -------------------------------------------------------------------
    // Index args
    // -------------------------------------------------------------------

    member private this.ParseIndexArg() : IndexArg =
        if this.Current().value = ":" then
            let cTok = tokens.[pos]
            pos <- pos + 1
            Colon(loc cTok.line cTok.col)
        else
            let startExpr = this.ParseExpr(0, false, false)  // colon hidden
            if this.Current().value = ":" then
                let colonTok = tokens.[pos]
                pos <- pos + 1
                let midExpr = this.ParseExpr(0, false, false)
                if this.Current().value = ":" then
                    // a:step:b — three-argument stepped range
                    pos <- pos + 1
                    let endExpr = this.ParseExpr(0, false, false)
                    SteppedRange(loc colonTok.line colonTok.col, startExpr, midExpr, endExpr)
                else
                    // a:b — two-argument range
                    Range(loc colonTok.line colonTok.col, startExpr, midExpr)
            else
                IndexExpr(loc startExpr.Line startExpr.Col, startExpr)

    member private this.ParseParenArgs() : IndexArg list =
        let args = System.Collections.Generic.List<IndexArg>()
        while this.Current().kind = TkNewline do pos <- pos + 1
        if this.Current().kind <> TkRParen && this.Current().kind <> TkRCurly then
            args.Add(this.ParseIndexArg())
            while this.Current().kind = TkComma do
                pos <- pos + 1  // eat comma
                while this.Current().kind = TkNewline do pos <- pos + 1
                args.Add(this.ParseIndexArg())
        while this.Current().kind = TkNewline do pos <- pos + 1
        args |> Seq.toList


// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Quick pre-scan: detect multi-function files that use end-less style.
/// In end-terminated files, end count = block openers + function count.
/// In end-less files, end count = block openers only (functions lack end).
/// If there are 2+ functions and fewer ends than openers + functions, it's end-less.
let private detectEndless (tokens: Token list) : bool =
    let mutable funcCount = 0
    let mutable openerCount = 0
    let mutable endCount = 0
    for tok in tokens do
        match tok.kind with
        | TkFunction -> funcCount <- funcCount + 1
        | TkIf | TkFor | TkParfor | TkWhile | TkSwitch | TkTry ->
            openerCount <- openerCount + 1
        | TkEnd -> endCount <- endCount + 1
        | _ -> ()
    funcCount >= 2 && endCount < openerCount + funcCount

let parseMATLAB (src: string) : Program * ParseDiagnostic list =
    let tokenList = Lexer.lex src
    if detectEndless tokenList then
        // Pre-scan detected end-less multi-function file; skip straight to endless mode
        let parser = MatlabParser(tokenList, true)
        (parser.ParseProgram(), parser.RecoveredErrors)
    else
        try
            // Try modern end-terminated functions first (most common)
            let parser = MatlabParser(tokenList, false)
            (parser.ParseProgram(), parser.RecoveredErrors)
        with
        | :? ParseError ->
            // Retry with endless-function mode (legacy MATLAB without end);
            // discard first parser's recovered errors (wrong parse tree)
            let parser = MatlabParser(tokenList, true)
            (parser.ParseProgram(), parser.RecoveredErrors)
