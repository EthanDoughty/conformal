module Lexer

open System
open System.Text.RegularExpressions

// Custom exception mirroring Python SyntaxError for lex errors.
exception LexError of string

// Token record matching the Python Token dataclass.
// col is 1-based; pos is 0-based character offset into the source string.
[<Struct>]
type Token =
    { kind: string
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
        """(?<DOTOP>\.\*|\./|\.\^|\.\')"""
        """(?<OP>==|~=|<=|>=|&&|\|\||[+\-*/<>()\[\]=,:\;@\\^&|~])"""
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
        """(?<DOTOP>\.\*|\./|\.\^|\.\')"""
        """(?<OP>==|~=|<=|>=|&&|\|\||[+\-*/<>()\[\]=,:\;@\\^&|~])"""
        """(?<DOT>\.)"""
        """(?<NEWLINE>\n)"""
        """(?<SKIP>[ \t]+)"""
        """(?<COMMENT>%[^\n]*)"""
        """(?<QUOTE>')"""
        """(?<CURLYBRACE>[{}])"""
        """(?<MISMATCH>.)"""
    ]

let private masterRe =
    Regex(masterPatternSimple, RegexOptions.Compiled)

// ---------------------------------------------------------------------------
// lex : string -> Token list
// ---------------------------------------------------------------------------
// Context-sensitive lexing for single quotes:
//   After ID / ) / ] / } / NUMBER / TRANSPOSE -> ' is TRANSPOSE
//   Otherwise -> ' starts a STRING
// Inside [] or {} (bracket_depth > 0), space before ' with those prev_kinds
// forces STRING (MATLAB matrix literal row semantics).

let lex (src: string) : Token list =
    // Normalize line endings.
    let src = src.Replace("\r\n", "\n").Replace("\r", "\n")

    let tokens = System.Collections.Generic.List<Token>()
    let mutable pos = 0
    let mutable line = 1
    let mutable lastNewlinePos = -1   // col 1 for pos 0 (1-based)
    let mutable prevKind: string = ""   // previous token kind ("" = start of file)
    let mutable bracketDepth = 0      // nesting depth of [ ] and { }
    let mutable sawSpace = false      // whitespace since last real token

    let inline makeToken kind value startPos =
        { kind = kind; value = value; pos = startPos; line = line; col = startPos - lastNewlinePos }

    while pos < src.Length do
        let m = masterRe.Match(src, pos, src.Length - pos)
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
            tokens.Add(makeToken "STRING" content startPos)
            prevKind <- "STRING"
            sawSpace <- false
            pos <- m.Index + m.Length

        | "NUMBER" ->
            tokens.Add(makeToken "NUMBER" value startPos)
            prevKind <- "NUMBER"
            sawSpace <- false
            pos <- m.Index + m.Length

        | "ID" ->
            let upper = value.ToUpperInvariant()
            if keywords.Contains(value) then
                tokens.Add(makeToken upper value startPos)
                prevKind <- upper
            else
                tokens.Add(makeToken "ID" value startPos)
                prevKind <- "ID"
            sawSpace <- false
            pos <- m.Index + m.Length

        | "DOTOP" ->
            tokens.Add(makeToken kind value startPos)
            prevKind <- kind
            sawSpace <- false
            pos <- m.Index + m.Length

        | "DOT" ->
            tokens.Add(makeToken "DOT" value startPos)
            prevKind <- "DOT"
            sawSpace <- false
            pos <- m.Index + m.Length

        | "CURLYBRACE" ->
            if value = "{" then
                bracketDepth <- bracketDepth + 1
            elif value = "}" then
                bracketDepth <- max 0 (bracketDepth - 1)
            tokens.Add(makeToken value value startPos)
            prevKind <- value
            sawSpace <- false
            pos <- m.Index + m.Length

        | "CONTINUATION" ->
            // Line continuation: ... (rest of line). Count the newline if present.
            if value.Contains('\n') then
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
                | "ID" | ")" | "]" | "}" | "NUMBER" | "TRANSPOSE" -> true
                | _ -> false
            // Inside brackets, space before ' overrides -> string
            let isTranspose =
                if isTranspose && bracketDepth > 0 && sawSpace then false
                else isTranspose

            if isTranspose then
                tokens.Add(makeToken "TRANSPOSE" value startPos)
                prevKind <- "TRANSPOSE"
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
                tokens.Add(makeToken "STRING" content startPos)
                prevKind <- "STRING"
                sawSpace <- false
                pos <- endPos + 1   // skip past closing quote

        | "OP" ->
            if value = "[" then bracketDepth <- bracketDepth + 1
            elif value = "]" then bracketDepth <- max 0 (bracketDepth - 1)
            tokens.Add(makeToken value value startPos)
            prevKind <- value
            sawSpace <- false
            pos <- m.Index + m.Length

        | "NEWLINE" ->
            tokens.Add(makeToken "NEWLINE" value startPos)
            line <- line + 1
            lastNewlinePos <- startPos
            prevKind <- "NEWLINE"
            sawSpace <- false
            pos <- m.Index + m.Length

        | "SKIP" | "COMMENT" ->
            sawSpace <- true
            pos <- m.Index + m.Length
            // Do NOT update prevKind for whitespace/comments.

        | "MISMATCH" ->
            raise (LexError("Unexpected character '" + value + "' at " + string startPos))

        | _ ->
            pos <- m.Index + m.Length

    // Append EOF sentinel.
    tokens.Add({ kind = "EOF"; value = ""; pos = src.Length; line = line; col = src.Length - lastNewlinePos })
    tokens |> Seq.toList
