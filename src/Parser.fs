module Parser

open System
open Lexer
open Ir

// Parse error exception, mirroring Python's ParseError.
exception ParseError of string


// ---------------------------------------------------------------------------
// extract_targets_from_tokens
// ---------------------------------------------------------------------------

let extractTargetsFromTokens (toks: Token list) : string list =
    if toks.IsEmpty then []
    else
        let arr = Array.ofList toks

        // Simple: ID =
        if arr.Length >= 2 && arr.[0].kind = "ID" && arr.[1].value = "=" then
            [arr.[0].value]

        // Function-style: ID( ... ) =
        elif arr.Length >= 3 && arr.[0].kind = "ID" && arr.[1].value = "(" then
            let mutable depth = 0
            let mutable result: string list = []
            let mutable i = 0
            while i < arr.Length && result.IsEmpty do
                if arr.[i].value = "(" then depth <- depth + 1
                elif arr.[i].value = ")" then
                    depth <- depth - 1
                    if depth = 0 then
                        if i + 1 < arr.Length && arr.[i + 1].value = "=" then
                            result <- [arr.[0].value]
                        i <- arr.Length  // break
                i <- i + 1
            result

        // Destructuring: [ ... ] =
        elif arr.Length >= 2 && arr.[0].value = "[" then
            let mutable depth = 0
            let mutable bracketEnd = -1
            let mutable i = 0
            while i < arr.Length && bracketEnd = -1 do
                if arr.[i].value = "[" then depth <- depth + 1
                elif arr.[i].value = "]" then
                    depth <- depth - 1
                    if depth = 0 then bracketEnd <- i
                i <- i + 1

            if bracketEnd > 0 && bracketEnd + 1 < arr.Length && arr.[bracketEnd + 1].value = "=" then
                let mutable valid = true
                for j in 1 .. bracketEnd - 1 do
                    let tok = arr.[j]
                    if tok.kind <> "ID" && tok.kind <> "NEWLINE" && tok.value <> "," && tok.value <> "~" && tok.value <> "." then
                        valid <- false
                if valid then
                    let targets = System.Collections.Generic.List<string>()
                    let mutable j = 1
                    while j < bracketEnd do
                        let tok = arr.[j]
                        if tok.kind = "ID" then
                            targets.Add(tok.value)
                            j <- j + 1
                            while j < bracketEnd && arr.[j].value = "." do
                                j <- j + 1
                                if j < bracketEnd && arr.[j].kind = "ID" then j <- j + 1
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

type MatlabParser(tokenList: Token list) =

    let tokens = Array.ofList tokenList
    let mutable pos = 0  // current token index

    // Endless-function pre-scan
    let detectEndlessFunctions () : bool =
        let mutable start = -1
        let mutable si = 0
        while si < tokens.Length && start = -1 do
            if tokens.[si].kind = "FUNCTION" then start <- si
            si <- si + 1
        if start = -1 then false
        else
            let blockOpeners = set ["IF"; "FOR"; "WHILE"; "SWITCH"; "TRY"]
            let mutable blockDepth = 1
            let mutable delimDepth = 0
            let mutable result = true
            let mutable finished = false
            let mutable idx = start + 1
            while idx < tokens.Length && not finished do
                let tok = tokens.[idx]
                if tok.kind = "EOF" then
                    result <- blockDepth >= 1
                    finished <- true
                else
                    if tok.value = "(" || tok.value = "[" || tok.value = "{" then
                        delimDepth <- delimDepth + 1
                    elif tok.value = ")" || tok.value = "]" || tok.value = "}" then
                        delimDepth <- max 0 (delimDepth - 1)
                    if delimDepth = 0 then
                        if Set.contains tok.kind blockOpeners then
                            blockDepth <- blockDepth + 1
                        elif tok.kind = "END" then
                            blockDepth <- blockDepth - 1
                            if blockDepth = 0 then
                                result <- false
                                finished <- true
                        elif tok.kind = "FUNCTION" then
                            if blockDepth >= 1 then blockDepth <- blockDepth + 1
                            else
                                result <- true
                                finished <- true
                idx <- idx + 1
            result

    let endlessFunctions = detectEndlessFunctions ()

    // Token helpers
    member private _.Current() = tokens.[pos]
    member private _.AtEnd() = tokens.[pos].kind = "EOF"

    member private _.Eat(expected: string) : Token =
        let tok = tokens.[pos]
        if tok.kind <> expected && tok.value <> expected then
            raise (ParseError("Expected " + expected + " at pos " + string tok.pos + ", found " + tok.kind + " '" + tok.value + "'"))
        pos <- pos + 1
        tok

    member private _.StartsExpr(tok: Token) : bool =
        tok.kind = "NUMBER" || tok.kind = "ID" || tok.kind = "STRING" || tok.kind = "END" ||
        tok.value = "(" || tok.value = "-" || tok.value = "+" || tok.value = "~" ||
        tok.value = "[" || tok.value = "{" || tok.value = "@"

    // Recovery: consume tokens until statement boundary
    member private this.RecoverToStmtBoundary(startLine: int, startCol: int) : Stmt =
        let consumed = System.Collections.Generic.List<Token>()
        let mutable depth = 0
        let mutable stop = false
        while not (this.AtEnd()) && not stop do
            let tok = this.Current()
            if depth = 0 then
                let isBlockEnd = tok.kind = "END" || tok.kind = "ELSE" || tok.kind = "ELSEIF" ||
                                 tok.kind = "CASE" || tok.kind = "OTHERWISE" || tok.kind = "CATCH"
                if isBlockEnd then
                    stop <- true
                elif tok.kind = "NEWLINE" then
                    pos <- pos + 1
                    stop <- true
                elif tok.kind = ";" || tok.value = ";" then
                    pos <- pos + 1
                    stop <- true
                else
                    if tok.value = "(" || tok.value = "[" || tok.value = "{" then depth <- depth + 1
                    elif tok.value = ")" || tok.value = "]" || tok.value = "}" then depth <- max 0 (depth - 1)
                    consumed.Add(tok)
                    pos <- pos + 1
            else
                if tok.value = "(" || tok.value = "[" || tok.value = "{" then depth <- depth + 1
                elif tok.value = ")" || tok.value = "]" || tok.value = "}" then depth <- max 0 (depth - 1)
                consumed.Add(tok)
                pos <- pos + 1
        let rawText = consumed |> Seq.map (fun t -> t.value) |> String.concat " "
        let targets = extractTargetsFromTokens (consumed |> Seq.toList)
        OpaqueStmt(startLine, startCol, targets, rawText)

    // Classdef consumer
    member private this.ConsumeClassdef() : Stmt =
        let startTok = this.Current()
        let line = startTok.line
        let col = startTok.col
        let mutable depth = 1
        pos <- pos + 1  // skip 'classdef'
        let blockOpeners = set ["IF"; "FOR"; "WHILE"; "SWITCH"; "TRY"; "FUNCTION"; "PARFOR"]
        let idBlockOpeners = set ["methods"; "properties"; "events"; "enumeration"]
        let mutable parenDepth = 0
        let mutable finished = false
        while not (this.AtEnd()) && not finished do
            let tok = this.Current()
            pos <- pos + 1
            if tok.value = "(" || tok.value = "[" || tok.value = "{" then
                parenDepth <- parenDepth + 1
            elif tok.value = ")" || tok.value = "]" || tok.value = "}" then
                parenDepth <- max 0 (parenDepth - 1)
            elif parenDepth = 0 then
                if tok.kind = "END" then
                    depth <- depth - 1
                    if depth = 0 then finished <- true
                elif Set.contains tok.kind blockOpeners then
                    depth <- depth + 1
                elif tok.kind = "ID" && Set.contains tok.value idBlockOpeners then
                    depth <- depth + 1
        while not (this.AtEnd()) && (this.Current().kind = "NEWLINE" || this.Current().kind = ";" || this.Current().value = ";") do
            pos <- pos + 1
        OpaqueStmt(line, col, [], "classdef")

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
                if cur.kind = "NEWLINE" then pos <- pos + 1
                elif cur.kind = ";" || cur.value = ";" then pos <- pos + 1
                else skipping <- false
            if not (this.AtEnd()) then
                let cur = this.Current()
                if cur.kind = "ID" && cur.value = "classdef" then
                    items.Add(this.ConsumeClassdef())
                elif cur.kind = "FUNCTION" then
                    items.Add(this.ParseFunctionDef())
                else
                    let savedPos = pos
                    items.Add(this.ParseStmt())
                    if pos = savedPos then pos <- pos + 1
        { body = items |> Seq.toList }

    // -------------------------------------------------------------------
    // Function definition
    // -------------------------------------------------------------------

    member private this.ParseFunctionDef() : Stmt =
        let funcTok = this.Eat("FUNCTION")
        let line = funcTok.line
        let col = funcTok.col
        let mutable outputVars: string list = []
        let mutable name = ""
        let mutable hasParens = true

        if this.Current().kind = "ID" then
            let lookahead = if pos + 1 < tokens.Length then tokens.[pos + 1] else tokens.[tokens.Length - 1]
            if lookahead.value = "(" then
                // Procedure form: function name(args)
                name <- this.Eat("ID").value
                this.Eat("(") |> ignore
            elif lookahead.value = "=" then
                // Single return: function result = name(args)
                let outVar = this.Eat("ID").value
                outputVars <- [outVar]
                this.Eat("=") |> ignore
                name <- this.Eat("ID").value
                if this.Current().value = "(" then this.Eat("(") |> ignore
                else hasParens <- false
            elif lookahead.kind = "NEWLINE" || lookahead.kind = "EOF" || lookahead.value = ";" ||
                 lookahead.kind = "FUNCTION" then
                // No-arg procedure
                name <- this.Eat("ID").value
                hasParens <- false
            else
                raise (ParseError("Expected '=' or '(' after function name at " + string (this.Current().pos)))
        elif this.Current().value = "[" then
            // Multiple outputs: function [a, b] = name(args)
            this.Eat("[") |> ignore
            if this.Current().value = "]" then
                this.Eat("]") |> ignore
            else
                outputVars <- [this.Eat("ID").value]
                while this.Current().value = "," || this.Current().kind = "ID" do
                    if this.Current().value = "," then this.Eat(",") |> ignore
                    outputVars <- outputVars @ [this.Eat("ID").value]
                this.Eat("]") |> ignore
            this.Eat("=") |> ignore
            name <- this.Eat("ID").value
            if this.Current().value = "(" then this.Eat("(") |> ignore
            else hasParens <- false
        else
            raise (ParseError("Expected function output or name at " + string (this.Current().pos)))

        // Parameters
        let mutable parms: string list = []
        if hasParens then
            let eatParam () =
                if this.Current().value = "~" then
                    this.Eat("~") |> ignore; "~"
                else this.Eat("ID").value
            if this.Current().value <> ")" then
                parms <- [eatParam ()]
                while this.Current().value = "," do
                    this.Eat(",") |> ignore
                    parms <- parms @ [eatParam ()]
            this.Eat(")") |> ignore

        // Skip newline/semicolon after signature
        if this.Current().kind = "NEWLINE" then this.Eat("NEWLINE") |> ignore
        elif this.Current().kind = ";" || this.Current().value = ";" then pos <- pos + 1

        // Body
        let body =
            if endlessFunctions then this.ParseBlock([| "FUNCTION" |])
            else
                let b = this.ParseBlock([| "END" |])
                this.Eat("END") |> ignore
                b

        FunctionDef(line, col, name, parms, outputVars, body)

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
            | "FOR"    -> this.ParseFor(false)
            | "PARFOR" -> this.ParseFor(true)
            | "GLOBAL" | "PERSISTENT" -> this.ParseGlobal()
            | "WHILE"  -> this.ParseWhile()
            | "IF"     -> this.ParseIf()
            | "SWITCH" -> this.ParseSwitch()
            | "TRY"    -> this.ParseTry()
            | "BREAK"  -> let t = this.Eat("BREAK") in Break(t.line, t.col)
            | "CONTINUE" -> let t = this.Eat("CONTINUE") in Continue(t.line, t.col)
            | "RETURN" -> let t = this.Eat("RETURN") in Return(t.line, t.col)
            | "FUNCTION" -> this.ParseFunctionDef()
            | "NEWLINE" ->
                this.Eat("NEWLINE") |> ignore
                ExprStmt(0, 0, Const(0, 0, 0.0))
            | _ ->
                let node = this.ParseSimpleStmt()
                let curKind = this.Current().kind
                let curVal  = this.Current().value
                if curKind <> "NEWLINE" && curKind <> "EOF" && curVal <> ";" then
                    pos <- startPos
                    this.RecoverToStmtBoundary(startLine, startCol)
                else
                    if this.Current().kind = "NEWLINE" then this.Eat("NEWLINE") |> ignore
                    elif this.Current().kind = ";" || this.Current().value = ";" then pos <- pos + 1
                    node
        with
        | :? ParseError ->
            pos <- startPos
            this.RecoverToStmtBoundary(startLine, startCol)

    // Bracket stmt: [a,b] = expr  or  matrix expression
    member private this.ParseBracketStmt() : Stmt =
        let savedPos = pos
        try
            this.Eat("[") |> ignore
            let eatTarget () =
                if this.Current().value = "~" then
                    this.Eat("~") |> ignore; "~"
                else
                    let mutable tname = this.Eat("ID").value
                    while this.Current().kind = "DOT" do
                        this.Eat("DOT") |> ignore
                        tname <- tname + "." + this.Eat("ID").value
                    tname
            let mutable targets = [eatTarget ()]
            while this.Current().value = "," || this.Current().kind = "ID" || this.Current().value = "~" do
                if this.Current().value = "," then this.Eat(",") |> ignore
                targets <- targets @ [eatTarget ()]
            this.Eat("]") |> ignore
            if this.Current().value = "=" then
                let eqTok = this.Eat("=")
                let expr = this.ParseExpr(0, false, true)
                AssignMulti(eqTok.line, eqTok.col, targets, expr)
            else
                pos <- savedPos
                let expr = this.ParseExpr(0, false, true)
                ExprStmt(expr.Line, expr.Col, expr)
        with
        | :? ParseError ->
            pos <- savedPos
            let expr = this.ParseExpr(0, false, true)
            ExprStmt(expr.Line, expr.Col, expr)

    // Greedy LHS accessor chain after an ID.
    // Returns list of kind+data: kind is "field"|"paren"|"curly", data is string|IndexArg list
    member private this.ParseLhsChain() : (string * obj) list =
        let chain = System.Collections.Generic.List<string * obj>()
        let mutable stop = false
        while not stop do
            let cur = this.Current()
            if cur.kind = "DOT" then
                this.Eat("DOT") |> ignore
                if this.Current().kind = "ID" then
                    chain.Add(("field", box (this.Eat("ID").value)))
                elif this.Current().value = "(" then
                    this.Eat("(") |> ignore
                    this.ParseExpr(0, false, true) |> ignore
                    this.Eat(")") |> ignore
                    chain.Add(("field", box "<dynamic>"))
                else
                    stop <- true
            elif cur.value = "(" then
                this.Eat("(") |> ignore
                let args = this.ParseParenArgs()
                this.Eat(")") |> ignore
                chain.Add(("paren", box args))
            elif cur.value = "{" then
                this.Eat("{") |> ignore
                let args = this.ParseParenArgs()
                this.Eat("}") |> ignore
                chain.Add(("curly", box args))
            else
                stop <- true
        chain |> Seq.toList

    member private _.ChainToExpr(idTok: Token, chain: (string * obj) list) : Expr =
        let mutable expr: Expr = Var(idTok.line, idTok.col, idTok.value)
        for (kind, data) in chain do
            match kind with
            | "field" -> expr <- FieldAccess(idTok.line, idTok.col, expr, data :?> string)
            | "paren" -> expr <- Apply(idTok.line, idTok.col, expr, data :?> IndexArg list)
            | "curly" -> expr <- CurlyApply(idTok.line, idTok.col, expr, data :?> IndexArg list)
            | _       -> ()
        expr

    member private _.ClassifyAssignment(idTok: Token, chain: (string * obj) list, eqTok: Token, rhs: Expr) : Stmt =
        let baseName = idTok.value

        if chain.IsEmpty then
            Assign(idTok.line, idTok.col, baseName, rhs)

        elif chain |> List.forall (fun (k,_) -> k = "field") then
            let fields = chain |> List.map (fun (_,d) -> d :?> string)
            StructAssign(eqTok.line, eqTok.col, baseName, fields, rhs)

        elif chain.Length = 1 && fst chain.[0] = "paren" then
            IndexAssign(eqTok.line, eqTok.col, baseName, snd chain.[0] :?> IndexArg list, rhs)

        elif chain.Length = 1 && fst chain.[0] = "curly" then
            CellAssign(eqTok.line, eqTok.col, baseName, snd chain.[0] :?> IndexArg list, rhs)

        elif (fst chain.[0] = "paren" || fst chain.[0] = "curly") && chain.Length > 1 &&
             chain.[1..] |> List.forall (fun (k,_) -> k = "field") then
            let indexKind = fst chain.[0]
            let indexArgs = snd chain.[0] :?> IndexArg list
            let fields = chain.[1..] |> List.map (fun (_,d) -> d :?> string)
            IndexStructAssign(eqTok.line, eqTok.col, baseName, indexArgs, indexKind, fields, rhs)

        else
            let indexPos = chain |> List.tryFindIndex (fun (k,_) -> k = "paren" || k = "curly")
            match indexPos with
            | Some ip ->
                let prefix = if ip > 0 then chain.[..ip-1] else []
                let suffix = if ip < chain.Length - 1 then chain.[ip+1..] else []
                if not prefix.IsEmpty && prefix |> List.forall (fun (k,_) -> k = "field") &&
                   not suffix.IsEmpty && suffix |> List.forall (fun (k,_) -> k = "field") then
                    FieldIndexAssign(
                        eqTok.line, eqTok.col, baseName,
                        prefix |> List.map (fun (_,d) -> d :?> string),
                        snd chain.[ip] :?> IndexArg list,
                        fst chain.[ip],
                        suffix |> List.map (fun (_,d) -> d :?> string),
                        rhs)
                elif chain.Length >= 2 && fst chain.[chain.Length-1] = "paren" &&
                     chain.[..chain.Length-2] |> List.forall (fun (k,_) -> k = "field") then
                    ExprStmt(eqTok.line, eqTok.col, rhs)
                elif chain.Length >= 2 && fst chain.[chain.Length-1] = "curly" &&
                     chain.[..chain.Length-2] |> List.forall (fun (k,_) -> k = "field") then
                    ExprStmt(eqTok.line, eqTok.col, rhs)
                else
                    OpaqueStmt(eqTok.line, eqTok.col, [baseName], "")
            | None ->
                OpaqueStmt(eqTok.line, eqTok.col, [baseName], "")

    member private this.ParseSimpleStmt() : Stmt =
        if this.Current().value = "[" then
            this.ParseBracketStmt()
        elif this.Current().kind = "ID" then
            let idTok = this.Eat("ID")
            let chain = this.ParseLhsChain()
            if this.Current().value = "=" then
                let eqTok = this.Eat("=")
                let rhs = this.ParseExpr(0, false, true)
                this.ClassifyAssignment(idTok, chain, eqTok, rhs)
            else
                let mutable expr = this.ChainToExpr(idTok, chain)
                expr <- this.ParsePostfix(expr)
                expr <- this.ParseExprRest(expr, 0, false, true)
                ExprStmt(expr.Line, expr.Col, expr)
        else
            let expr = this.ParseExpr(0, false, true)
            ExprStmt(expr.Line, expr.Col, expr)

    // -------------------------------------------------------------------
    // Control flow
    // -------------------------------------------------------------------

    member private this.ParseFor(isParfor: bool) : Stmt =
        this.Eat(if isParfor then "PARFOR" else "FOR") |> ignore
        let varTok = this.Eat("ID")
        this.Eat("=") |> ignore
        let it = this.ParseExpr(0, false, true)
        let body = this.ParseBlock([| "END" |])
        this.Eat("END") |> ignore
        For(it.Line, it.Col, varTok.value, it, body)

    member private this.ParseGlobal() : Stmt =
        let kwTok = tokens.[pos]
        pos <- pos + 1
        let mutable varNames: string list = []
        while this.Current().kind = "ID" do
            varNames <- varNames @ [this.Eat("ID").value]
        let rawText = "global " + String.concat " " varNames
        OpaqueStmt(kwTok.line, kwTok.col, varNames, rawText)

    member private this.ParseWhile() : Stmt =
        this.Eat("WHILE") |> ignore
        let cond = this.ParseExpr(0, false, true)
        let body = this.ParseBlock([| "END" |])
        this.Eat("END") |> ignore
        While(cond.Line, cond.Col, cond, body)

    member private this.ParseIf() : Stmt =
        this.Eat("IF") |> ignore
        let cond = this.ParseExpr(0, false, true)
        let thenBody = this.ParseBlock([| "ELSE"; "ELSEIF"; "END" |])
        let mutable elseifs: (Expr * Stmt list) list = []
        while this.Current().kind = "ELSEIF" do
            this.Eat("ELSEIF") |> ignore
            let elifCond = this.ParseExpr(0, false, true)
            let elifBody = this.ParseBlock([| "ELSE"; "ELSEIF"; "END" |])
            elseifs <- elseifs @ [(elifCond, elifBody)]
        let skipStmt = ExprStmt(0, 0, Const(0, 0, 0.0))
        let mutable elseBody = [skipStmt]
        if this.Current().kind = "ELSE" then
            this.Eat("ELSE") |> ignore
            elseBody <- this.ParseBlock([| "END" |])
        this.Eat("END") |> ignore
        if elseifs.IsEmpty then
            If(cond.Line, cond.Col, cond, thenBody, elseBody)
        else
            let conditions = cond :: (elseifs |> List.map fst)
            let bodies = thenBody :: (elseifs |> List.map snd)
            IfChain(cond.Line, cond.Col, conditions, bodies, elseBody)

    member private this.ParseSwitch() : Stmt =
        this.Eat("SWITCH") |> ignore
        let expr = this.ParseExpr(0, false, true)
        if this.Current().kind = "NEWLINE" then this.Eat("NEWLINE") |> ignore
        let mutable cases: (Expr * Stmt list) list = []
        while this.Current().kind = "CASE" do
            this.Eat("CASE") |> ignore
            let caseVal = this.ParseExpr(0, false, true)
            let caseBody = this.ParseBlock([| "CASE"; "OTHERWISE"; "END" |])
            cases <- cases @ [(caseVal, caseBody)]
        let skipStmt = ExprStmt(0, 0, Const(0, 0, 0.0))
        let mutable otherwiseBody = [skipStmt]
        if this.Current().kind = "OTHERWISE" then
            this.Eat("OTHERWISE") |> ignore
            otherwiseBody <- this.ParseBlock([| "END" |])
        this.Eat("END") |> ignore
        Switch(expr.Line, expr.Col, expr, cases, otherwiseBody)

    member private this.ParseTry() : Stmt =
        this.Eat("TRY") |> ignore
        let tryBody = this.ParseBlock([| "CATCH"; "END" |])
        let skipStmt = ExprStmt(0, 0, Const(0, 0, 0.0))
        let mutable catchBody = [skipStmt]
        if this.Current().kind = "CATCH" then
            this.Eat("CATCH") |> ignore
            if this.Current().kind = "ID" then this.Eat("ID") |> ignore
            catchBody <- this.ParseBlock([| "END" |])
        this.Eat("END") |> ignore
        let line, col =
            match tryBody with
            | h :: _ -> h.Line, h.Col
            | [] -> 0, 0
        Try(line, col, tryBody, catchBody)

    member private this.ParseBlock(untilKinds: string[]) : Stmt list =
        if this.Current().kind = "NEWLINE" then this.Eat("NEWLINE") |> ignore
        let stmts = System.Collections.Generic.List<Stmt>()
        while not (this.AtEnd()) && not (Array.contains (this.Current().kind) untilKinds) do
            let savedPos = pos
            stmts.Add(this.ParseStmt())
            if pos = savedPos then pos <- pos + 1
        if stmts.Count = 0 then
            [ExprStmt(0, 0, Const(0, 0, 0.0))]
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
        match tok.value with
        | "+" ->
            pos <- pos + 1
            // unary + is identity
            this.ParseExpr(precedenceTable.["+"], matrixContext, colonVisible)

        | "-" ->
            let minusTok = tokens.[pos]
            pos <- pos + 1
            let operand = this.ParseExpr(precedenceTable.["-"], matrixContext, colonVisible)
            Neg(minusTok.line, minusTok.col, operand)

        | "~" ->
            let notTok = tokens.[pos]
            pos <- pos + 1
            let operand = this.ParseExpr(precedenceTable.["+"], matrixContext, colonVisible)
            Not(notTok.line, notTok.col, operand)

        | "@" ->
            let atTok = tokens.[pos]
            pos <- pos + 1
            let next = this.Current()
            if next.value = "(" then
                this.Eat("(") |> ignore
                let mutable parms: string list = []
                if this.Current().value <> ")" then
                    parms <- [this.Eat("ID").value]
                    while this.Current().value = "," do
                        this.Eat(",") |> ignore
                        parms <- parms @ [this.Eat("ID").value]
                this.Eat(")") |> ignore
                let body = this.ParseExpr(0, false, true)
                Lambda(atTok.line, atTok.col, parms, body)
            elif next.kind = "ID" then
                let name = this.Eat("ID").value
                FuncHandle(atTok.line, atTok.col, name)
            else
                raise (ParseError("Expected '(' or function name after '@' at " + string next.pos))

        | "[" ->
            let ml = this.ParseMatrixLiteral()
            this.ParsePostfix(ml)

        | "{" ->
            let cl = this.ParseCellLiteral()
            this.ParsePostfix(cl)

        | "(" ->
            this.Eat("(") |> ignore
            let inner = this.ParseExpr(0, false, true)
            this.Eat(")") |> ignore
            this.ParsePostfix(inner)

        | _ ->
            match tok.kind with
            | "NUMBER" ->
                let numTok = tokens.[pos]
                pos <- pos + 1
                Const(numTok.line, numTok.col, float numTok.value)

            | "STRING" ->
                let strTok = tokens.[pos]
                pos <- pos + 1
                StringLit(strTok.line, strTok.col, strTok.value)

            | "ID" ->
                let idTok = tokens.[pos]
                pos <- pos + 1
                let left: Expr = Var(idTok.line, idTok.col, idTok.value)
                this.ParsePostfix(left)

            | "END" ->
                let endTok = tokens.[pos]
                pos <- pos + 1
                End(endTok.line, endTok.col)

            | _ ->
                raise (ParseError("Unexpected token " + tok.kind + " '" + tok.value + "' in expression at " + string tok.pos))

    member private this.ParsePostfix(initial: Expr) : Expr =
        let mutable left = initial
        let mutable stop = false
        while not stop do
            let tok = this.Current()
            if tok.value = "(" then
                let lparenTok = tokens.[pos]
                pos <- pos + 1
                let args = this.ParseParenArgs()
                this.Eat(")") |> ignore
                left <- Apply(lparenTok.line, lparenTok.col, left, args)

            elif tok.value = "{" then
                let lcurlyTok = tokens.[pos]
                pos <- pos + 1
                let args = this.ParseParenArgs()
                this.Eat("}") |> ignore
                left <- CurlyApply(lcurlyTok.line, lcurlyTok.col, left, args)

            elif tok.kind = "TRANSPOSE" then
                let tTok = tokens.[pos]
                pos <- pos + 1
                left <- Transpose(tTok.line, tTok.col, left)

            elif tok.kind = "DOTOP" && tok.value = ".'" then
                let tTok = tokens.[pos]
                pos <- pos + 1
                left <- Transpose(tTok.line, tTok.col, left)

            elif tok.kind = "DOT" then
                let dotTok = tokens.[pos]
                pos <- pos + 1
                if this.Current().kind = "ID" then
                    let fieldName = this.Eat("ID").value
                    left <- FieldAccess(dotTok.line, dotTok.col, left, fieldName)
                elif this.Current().value = "(" then
                    this.Eat("(") |> ignore
                    this.ParseExpr(0, false, true) |> ignore
                    this.Eat(")") |> ignore
                    left <- FieldAccess(dotTok.line, dotTok.col, left, "<dynamic>")
                else
                    raise (ParseError("Expected field name after '.' at " + string tok.pos))

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
            if not (precedenceTable.TryGetValue(op, &prec)) then
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
                    left <- BinOp(opTok.line, opTok.col, op, left, right)
        left

    // -------------------------------------------------------------------
    // Matrix / cell literals
    // -------------------------------------------------------------------

    member private this.ParseDelimitedRows(endToken: string) : int * int * Expr list list =
        let cur = this.Current()
        let line = cur.line
        let col = cur.col

        if this.Current().value = endToken then
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
                    if tok.value = "," then
                        pos <- pos + 1
                        row.Add(this.ParseExpr(0, true, true))
                    elif tok.value = ";" || tok.value = endToken || tok.kind = "NEWLINE" || tok.kind = "EOF" then
                        innerStop <- true
                    elif this.StartsExpr(tok) then
                        row.Add(this.ParseExpr(0, true, true))
                    else
                        innerStop <- true
                rows.Add(row |> Seq.toList)

                if this.Current().value = endToken then
                    outerStop <- true
                elif this.Current().value = ";" then
                    pos <- pos + 1
                    if this.Current().value = endToken then outerStop <- true
                elif this.Current().kind = "NEWLINE" then
                    pos <- pos + 1
                    if this.Current().value = endToken then outerStop <- true
                else
                    let tok = this.Current()
                    raise (ParseError("Unexpected token " + tok.kind + " '" + tok.value + "' in literal at " + string tok.pos))
            (line, col, rows |> Seq.toList)

    member private this.ParseMatrixLiteral() : Expr =
        let lbrack = this.Eat("[")
        let line, col, rows = this.ParseDelimitedRows("]")
        this.Eat("]") |> ignore
        let l, c = if rows.IsEmpty then lbrack.line, lbrack.col else line, col
        MatrixLit(l, c, rows)

    member private this.ParseCellLiteral() : Expr =
        let lcurly = this.Eat("{")
        let line, col, rows = this.ParseDelimitedRows("}")
        this.Eat("}") |> ignore
        let l, c = if rows.IsEmpty then lcurly.line, lcurly.col else line, col
        CellLit(l, c, rows)

    // -------------------------------------------------------------------
    // Index args
    // -------------------------------------------------------------------

    member private this.ParseIndexArg() : IndexArg =
        if this.Current().value = ":" then
            let cTok = tokens.[pos]
            pos <- pos + 1
            Colon(cTok.line, cTok.col)
        else
            let startExpr = this.ParseExpr(0, false, false)  // colon hidden
            if this.Current().value = ":" then
                let colonTok = tokens.[pos]
                pos <- pos + 1
                let rangeEnd = this.ParseExpr(0, false, false)
                Range(colonTok.line, colonTok.col, startExpr, rangeEnd)
            else
                IndexExpr(startExpr.Line, startExpr.Col, startExpr)

    member private this.ParseParenArgs() : IndexArg list =
        let args = System.Collections.Generic.List<IndexArg>()
        while this.Current().kind = "NEWLINE" do pos <- pos + 1
        if this.Current().value <> ")" && this.Current().value <> "}" then
            args.Add(this.ParseIndexArg())
            while this.Current().value = "," do
                pos <- pos + 1  // eat comma
                while this.Current().kind = "NEWLINE" do pos <- pos + 1
                args.Add(this.ParseIndexArg())
        while this.Current().kind = "NEWLINE" do pos <- pos + 1
        args |> Seq.toList


// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

let parseMATLAB (src: string) : Program =
    let tokenList = Lexer.lex src
    let parser = MatlabParser(tokenList)
    parser.ParseProgram()
