module LspHover

open System.Text.RegularExpressions
open Ionide.LanguageServerProtocol.Types
open Shapes
open Context

// ---------------------------------------------------------------------------
// getHover: hover provider showing inferred shapes.
// Port of lsp/hover.py get_hover().
// ---------------------------------------------------------------------------

/// getHover: return hover information for identifier at (line, character).
/// line and character are 0-based (LSP convention).
let getHover
    (env: Env.Env)
    (source: string)
    (line: int)
    (character: int)
    (functionRegistry: System.Collections.Generic.Dictionary<string, FunctionSignature>)
    (knownBuiltins: Set<string>)
    (externalFunctions: System.Collections.Generic.Dictionary<string, ExternalSignature>)
    : Hover option =

    let lines = source.Split('\n')
    if line < 0 || line >= lines.Length then None
    else

    let lineText = lines.[line]
    if character < 0 || character > lineText.Length then None
    else

    // Extract identifier at cursor position using regex
    let identPattern = Regex(@"[A-Za-z_]\w*")
    let mutable word = ""
    let mutable matchStart = 0
    let mutable matchEnd   = 0
    let mutable found = false

    let mMatches = identPattern.Matches(lineText)
    let mutable idx = 0
    while not found && idx < mMatches.Count do
        let m = mMatches.[idx]
        let s, e = m.Index, m.Index + m.Length
        if s <= character && character < e then
            word       <- m.Value
            matchStart <- s
            matchEnd   <- e
            found      <- true
        idx <- idx + 1

    if not found then None
    else

    // Hover range spanning the matched identifier (uint32 for LSP positions)
    let hoverRange : Range = {
        Start = { Line = uint32 line; Character = uint32 matchStart }
        End   = { Line = uint32 line; Character = uint32 matchEnd }
    }

    let makeHover (text: string) : Hover =
        {
            Contents = U3.C1 { Kind = MarkupKind.Markdown; Value = text }
            Range    = Some hoverRange
        }

    // 1. Look up in environment (variable shape)
    let shape = Env.Env.get env word
    if shape <> Bottom && shape <> UnknownShape then
        let shapeStr = Shapes.shapeToString shape
        Some (makeHover ("(conformal) `" + word + "`: `" + shapeStr + "`"))
    else

    // 2. Check function_registry (same-file user-defined functions)
    match functionRegistry.TryGetValue(word) with
    | true, sig_ ->
        let paramsStr  = sig_.parms      |> String.concat ", "
        let outputsStr = sig_.outputVars |> String.concat ", "
        Some (makeHover ("(function) `" + word + "(" + paramsStr + ") -> [" + outputsStr + "]`"))
    | _ ->

    // 3. Check KNOWN_BUILTINS
    if Set.contains word knownBuiltins then
        Some (makeHover ("(builtin) `" + word + "`"))
    else

    // 4. Check external_functions (workspace)
    match externalFunctions.TryGetValue(word) with
    | true, extSig ->
        let filename = System.IO.Path.GetFileName(extSig.filename)
        Some (makeHover ("(external) `" + word + "` from `" + filename + "`"))
    | _ -> None
