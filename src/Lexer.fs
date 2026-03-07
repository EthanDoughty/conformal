module Lexer

open System
open System.Text.RegularExpressions

// Custom exception mirroring Python SyntaxError for lex errors.
exception LexError of string

type TokenKind =
    | TkId | TkNumber | TkString
    | TkLParen | TkRParen | TkLBracket | TkRBracket | TkLCurly | TkRCurly
    | TkComma | TkSemicolon | TkDot
    | TkOp | TkDotOp | TkTranspose
    | TkNewline | TkEof | TkShellEscape
    | TkIf | TkElse | TkElseif | TkEnd
    | TkFor | TkParfor | TkWhile
    | TkSwitch | TkCase | TkOtherwise
    | TkTry | TkCatch
    | TkFunction | TkReturn | TkBreak | TkContinue
    | TkGlobal | TkPersistent

let tokenKindName = function
    | TkId -> "ID" | TkNumber -> "NUMBER" | TkString -> "STRING"
    | TkLParen -> "(" | TkRParen -> ")" | TkLBracket -> "[" | TkRBracket -> "]"
    | TkLCurly -> "{" | TkRCurly -> "}" | TkComma -> "," | TkSemicolon -> ";"
    | TkDot -> "." | TkOp -> "OP" | TkDotOp -> "DOTOP" | TkTranspose -> "TRANSPOSE"
    | TkNewline -> "NEWLINE" | TkEof -> "EOF" | TkShellEscape -> "SHELL_ESCAPE"
    | TkIf -> "IF" | TkElse -> "ELSE" | TkElseif -> "ELSEIF" | TkEnd -> "END"
    | TkFor -> "FOR" | TkParfor -> "PARFOR" | TkWhile -> "WHILE"
    | TkSwitch -> "SWITCH" | TkCase -> "CASE" | TkOtherwise -> "OTHERWISE"
    | TkTry -> "TRY" | TkCatch -> "CATCH" | TkFunction -> "FUNCTION"
    | TkReturn -> "RETURN" | TkBreak -> "BREAK" | TkContinue -> "CONTINUE"
    | TkGlobal -> "GLOBAL" | TkPersistent -> "PERSISTENT"

// Token record matching the Python Token dataclass.
// col is 1-based; pos is 0-based character offset into the source string.
type Token =
    { kind: TokenKind
      value: string
      pos: int
      line: int
      col: int }

// MATLAB keywords (same set as Python KEYWORDS).
let keywords: Set<string> =
    Set.ofList [
        "for"; "parfor"; "while"; "if"; "else"; "elseif"; "end"
        "switch"; "case"; "otherwise"
        "try"; "catch"
        "break"; "continue"
        "function"; "return"
        "global"; "persistent"
    ]

let private keywordToKind (kw: string) : TokenKind =
    match kw.ToUpperInvariant() with
    | "FOR" -> TkFor | "PARFOR" -> TkParfor | "WHILE" -> TkWhile
    | "IF" -> TkIf | "ELSE" -> TkElse | "ELSEIF" -> TkElseif | "END" -> TkEnd
    | "SWITCH" -> TkSwitch | "CASE" -> TkCase | "OTHERWISE" -> TkOtherwise
    | "TRY" -> TkTry | "CATCH" -> TkCatch
    | "BREAK" -> TkBreak | "CONTINUE" -> TkContinue
    | "FUNCTION" -> TkFunction | "RETURN" -> TkReturn
    | "GLOBAL" -> TkGlobal | "PERSISTENT" -> TkPersistent
    | _ -> TkId

let private opValueToKind (v: string) : TokenKind =
    match v with
    | "(" -> TkLParen | ")" -> TkRParen
    | "[" -> TkLBracket | "]" -> TkRBracket
    | "," -> TkComma | ";" -> TkSemicolon
    | _ -> TkOp

// ---------------------------------------------------------------------------
// Token patterns – ordered by priority matching TOKEN_SPEC in lexer.py.
// We build one alternation regex with named groups.
// ---------------------------------------------------------------------------

// Named pattern fragments used in the master regex.
// Python TOKEN_SPEC order:
//   DQSTRING, NUMBER, ID, CONTINUATION, DOTOP, OP, DOT, NEWLINE, SKIP, COMMENT, QUOTE, CURLYBRACE, MISMATCH
// We replicate them as named groups; the first match wins.
//
// Note: F# Regex named groups use the same (?P<name>...) syntax as Python OR (?<name>...).
// .NET uses (?<name>...) for named groups.

let private masterPattern =
    String.concat "|" [
        """(?<DQSTRING>"(?:[^"]|"")*")"""
        // Scientific notation must come before plain number to match 1e5 before 1.
        // Also handle leading-dot floats like .5e+2 and .5
        """(?<NUMBER>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|(?<=\s|^|[+\-*/<>=,;:\[\]({~&|])\.\d+(?:[eE][+-]?\d+)?)"""
        """(?<ID>[A-Za-z_]\w*)"""
        """(?<CONTINUATION>\.\.\.[^\n]*\n?)"""
        """(?<DOTOP>\.\*|\./|\.\^|\.')"""
        """(?<OP>==|~=|<=|>=|&&|\|\||[+\-*/<>()\[\]=,:;@\\^&|~?])"""
        """(?<DOT>\.)"""
        """(?<NEWLINE>\n)"""
        """(?<SKIP>[ \t]+)"""
        """(?<COMMENT>%[^\n]*)"""
        """(?<QUOTE>')"""
        """(?<CURLYBRACE>[{}])"""
        """(?<MISMATCH>.)"""
    ]

// The leading-dot number pattern with lookbehind is awkward in this context.
// Let's simplify: we handle the leading-dot float as part of the DOTOP/DOT disambiguation
// in the main loop by checking after a DOT match. Actually the Python lexer relies on the
// regex ORDER — CONTINUATION before DOTOP before NUMBER. The key insight is that
// in Python, the NUMBER pattern r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?" does NOT match .5 —
// the Python lexer just doesn't support leading-dot numbers (they get lexed as DOT + NUMBER).
// Looking at lexer.py: NUMBER = r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?" — starts with \d+, no leading dot.
// So we should match the Python behavior exactly: no leading-dot support.

let private masterPatternSimple =
    String.concat "|" [
        """(?<DQSTRING>"(?:[^"]|"")*")"""
        """(?<NUMBER>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"""
        """(?<ID>[A-Za-z_]\w*)"""
        """(?<CONTINUATION>\.\.\.[^\n]*\n?)"""
        """(?<DOTOP>\.\*|\./|\.\^|\.')"""
        """(?<OP>==|~=|<=|>=|&&|\|\||[+\-*/<>()\[\]=,:;@\\^&|~?])"""
        """(?<DOT>\.)"""
        """(?<NEWLINE>\n)"""
        """(?<SKIP>[ \t]+)"""
        """(?<COMMENT>%[^\n]*)"""
        """(?<QUOTE>')"""
        """(?<CURLYBRACE>[{}])"""
        """(?<MISMATCH>.)"""
    ]

#if FABLE_COMPILER
let private masterRe = Regex(masterPatternSimple)
#else
let private masterRe = Regex(masterPatternSimple, RegexOptions.Compiled)
#endif

// Known MATLAB command-syntax functions: when these appear as an ID at statement
// level followed by ', the quote starts a string (not transpose).
// e.g., warning 'text', error 'msg', xlabel 'Label'
let private commandSyntaxNames =
    Set.ofList [
        "error"; "warning"; "disp"; "fprintf"; "sprintf"; "printf"
        "xlabel"; "ylabel"; "zlabel"; "title"; "sgtitle"; "legend"
        "cd"; "load"; "save"; "type"; "help"; "doc"; "which"; "what"
        "format"; "dbstop"; "dbclear"; "dbcont"; "dbtype"; "import"
        "mkdir"; "rmdir"; "delete"; "copyfile"; "movefile"
        "addpath"; "rmpath"; "rehash"; "mex"
    ]

// ---------------------------------------------------------------------------
// lex : string -> Token list
// ---------------------------------------------------------------------------
// Context-sensitive lexing for single quotes:
//   After ID / ) / ] / } / NUMBER / TRANSPOSE -> ' is TRANSPOSE
//   Otherwise -> ' starts a STRING
// Inside [] or {} (bracket_depth > 0), space before ' with those prev_kinds
// forces STRING (MATLAB matrix literal row semantics).
// After known command-syntax names with space -> ' starts a STRING.

let lex (src: string) : Token list =
    // Normalize line endings.
    let src = src.Replace("\r\n", "\n").Replace("\r", "\n")
    // Strip UTF-8 BOM if present.
    let src = if src.Length > 0 && src.[0] = '\uFEFF' then src.[1..] else src

    let tokens = System.Collections.Generic.List<Token>()
    let mutable pos = 0
    let mutable line = 1
    let mutable lastNewlinePos = -1   // col 1 for pos 0 (1-based)
    let mutable prevKind: TokenKind = TkNewline   // start-of-file acts as start-of-line
    let mutable prevValue: string = ""            // previous token's value (for command-syntax detection)
    let mutable bracketDepth = 0      // nesting depth of [ ] and { }
    let mutable sawSpace = false      // whitespace since last real token

    let inline makeToken kind value startPos =
        { kind = kind; value = value; pos = startPos; line = line; col = startPos - lastNewlinePos }

    while pos < src.Length do
#if FABLE_COMPILER
        let m = masterRe.Match(src, pos)
#else
        let m = masterRe.Match(src, pos, src.Length - pos)
#endif
        if not m.Success then
            raise (LexError("Unexpected character '" + string src.[pos] + "' at position " + string pos))

        // Which named group matched?
        let kind =
            if   m.Groups.["DQSTRING"].Success    then "DQSTRING"
            elif m.Groups.["NUMBER"].Success       then "NUMBER"
            elif m.Groups.["ID"].Success           then "ID"
            elif m.Groups.["CONTINUATION"].Success then "CONTINUATION"
            elif m.Groups.["DOTOP"].Success        then "DOTOP"
            elif m.Groups.["OP"].Success           then "OP"
            elif m.Groups.["DOT"].Success          then "DOT"
            elif m.Groups.["NEWLINE"].Success      then "NEWLINE"
            elif m.Groups.["SKIP"].Success         then "SKIP"
            elif m.Groups.["COMMENT"].Success      then "COMMENT"
            elif m.Groups.["QUOTE"].Success        then "QUOTE"
            elif m.Groups.["CURLYBRACE"].Success   then "CURLYBRACE"
            else                                        "MISMATCH"

        let value = m.Value
        let startPos = pos

        // Guard: the match must start exactly at pos (not further ahead).
        // Regex.Match with startAt can find a match after pos; we must enforce anchoring.
        if m.Index <> pos then
            raise (LexError("Unexpected character '" + string src.[pos] + "' at position " + string pos))

        match kind with
        | "DQSTRING" ->
            // Strip outer quotes; replace "" with "
            let content = value.[1 .. value.Length - 2].Replace("\"\"", "\"")
            tokens.Add(makeToken TkString content startPos)
            prevKind <- TkString
            sawSpace <- false
            pos <- m.Index + m.Length

        | "NUMBER" ->
            tokens.Add(makeToken TkNumber value startPos)
            prevKind <- TkNumber
            sawSpace <- false
            pos <- m.Index + m.Length

        | "ID" ->
            let upper = value.ToUpperInvariant()
            if keywords.Contains(value) then
                let tk = keywordToKind upper
                tokens.Add(makeToken tk value startPos)
                prevKind <- tk
            else
                tokens.Add(makeToken TkId value startPos)
                prevKind <- TkId
            prevValue <- value
            sawSpace <- false
            pos <- m.Index + m.Length

        | "DOTOP" ->
            tokens.Add(makeToken TkDotOp value startPos)
            prevKind <- TkDotOp
            sawSpace <- false
            pos <- m.Index + m.Length

        | "DOT" ->
            tokens.Add(makeToken TkDot value startPos)
            prevKind <- TkDot
            sawSpace <- false
            pos <- m.Index + m.Length

        | "CURLYBRACE" ->
            let tk = if value = "{" then TkLCurly else TkRCurly
            if value = "{" then
                bracketDepth <- bracketDepth + 1
            elif value = "}" then
                bracketDepth <- max 0 (bracketDepth - 1)
            tokens.Add(makeToken tk value startPos)
            prevKind <- tk
            sawSpace <- false
            pos <- m.Index + m.Length

        | "CONTINUATION" ->
            // Line continuation: ... (rest of line). Count the newline if present.
            if value.Contains("\n") then
                line <- line + 1
                let nlIdx = startPos + value.LastIndexOf('\n')
                lastNewlinePos <- nlIdx
            pos <- m.Index + m.Length
            // Do NOT update prevKind — preserves transpose/string context across lines.

        | "QUOTE" ->
            // Context-sensitive: transpose vs string start.
            // Matches Python: is_transpose if prev_kind in {ID, ), ], }, NUMBER, TRANSPOSE}
            let isTranspose =
                match prevKind with
                | TkId | TkRParen | TkRBracket | TkRCurly | TkNumber | TkTranspose -> true
                | _ -> false
            // Inside brackets, space before ' overrides -> string
            let isTranspose =
                if isTranspose && bracketDepth > 0 && sawSpace then false
                else isTranspose
            // Command syntax: known function names followed by space then ' -> string
            // e.g., warning 'text', xlabel 'Label'
            let isTranspose =
                if isTranspose && prevKind = TkId && sawSpace && commandSyntaxNames.Contains(prevValue) then false
                else isTranspose

            if isTranspose then
                tokens.Add(makeToken TkTranspose value startPos)
                prevKind <- TkTranspose
                sawSpace <- false
                pos <- m.Index + m.Length
            else
                // String: scan ahead for matching closing quote (no '' escaping in Python version).
                let mutable endPos = startPos + 1
                let mutable found = false
                while endPos < src.Length && not found do
                    if src.[endPos] = '\'' then
                        found <- true
                    elif src.[endPos] = '\n' then
                        raise (LexError("Unterminated string at line " + string line + ", pos " + string startPos))
                    else
                        endPos <- endPos + 1
                if not found then
                    raise (LexError("Unterminated string at line " + string line + ", pos " + string startPos))
                let content = src.[startPos + 1 .. endPos - 1]
                tokens.Add(makeToken TkString content startPos)
                prevKind <- TkString
                sawSpace <- false
                pos <- endPos + 1   // skip past closing quote

        | "OP" ->
            if value = "[" then bracketDepth <- bracketDepth + 1
            elif value = "]" then bracketDepth <- max 0 (bracketDepth - 1)
            let tk = opValueToKind value
            tokens.Add(makeToken tk value startPos)
            prevKind <- tk
            sawSpace <- false
            pos <- m.Index + m.Length

        | "NEWLINE" ->
            tokens.Add(makeToken TkNewline value startPos)
            line <- line + 1
            lastNewlinePos <- startPos
            prevKind <- TkNewline
            sawSpace <- false
            pos <- m.Index + m.Length

        | "SKIP" | "COMMENT" ->
            sawSpace <- true
            pos <- m.Index + m.Length
            // Do NOT update prevKind for whitespace/comments.

        | "MISMATCH" ->
            // MATLAB shell escape: !command runs the rest of the line as an OS command.
            // Valid at statement level (after NEWLINE, ";", or start-of-file).
            if value = "!" && (prevKind = TkNewline || prevKind = TkSemicolon) then
                let mutable endPos = startPos + 1
                while endPos < src.Length && src.[endPos] <> '\n' do
                    endPos <- endPos + 1
                let cmd = src.[startPos + 1 .. endPos - 1].Trim()
                tokens.Add(makeToken TkShellEscape cmd startPos)
                prevKind <- TkShellEscape
                sawSpace <- false
                pos <- endPos  // stop before \n so NEWLINE is lexed normally
            else
                raise (LexError("Unexpected character '" + value + "' at " + string startPos))

        | _ ->
            pos <- m.Index + m.Length

    // Append EOF sentinel.
    tokens.Add({ kind = TkEof; value = ""; pos = src.Length; line = line; col = src.Length - lastNewlinePos })
    tokens |> Seq.toList
