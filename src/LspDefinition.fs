module LspDefinition

open System.Text.RegularExpressions
open Ionide.LanguageServerProtocol.Types
open Context

// ---------------------------------------------------------------------------
// getDefinition: go-to-definition provider for user-defined functions.
// Returns Location option for Ctrl+Click / F12 navigation.
// ---------------------------------------------------------------------------

/// getDefinition: return definition location for identifier at (line, character).
/// line and character are 0-based (LSP convention).
let getDefinition
    (source: string)
    (line: int)
    (character: int)
    (uri: string)
    (functionRegistry: System.Collections.Generic.Dictionary<string, FunctionSignature>)
    (externalFunctions: System.Collections.Generic.Dictionary<string, ExternalSignature>)
    : Location option =

    let lines = source.Split('\n')
    if line < 0 || line >= lines.Length then None
    else

    let lineText = lines.[line]
    if character < 0 || character > lineText.Length then None
    else

    // Extract identifier at cursor position using regex (same logic as LspHover)
    let identPattern = Regex(@"[A-Za-z_]\w*")
    let mutable word = ""
    let mutable found = false

    let mMatches = identPattern.Matches(lineText)
    let mutable idx = 0
    while not found && idx < mMatches.Count do
        let m = mMatches.[idx]
        let s, e = m.Index, m.Index + m.Length
        if s <= character && character < e then
            word  <- m.Value
            found <- true
        idx <- idx + 1

    if not found then None
    else

    // 1. Check functionRegistry (same-file user-defined functions)
    match functionRegistry.TryGetValue(word) with
    | true, sig_ ->
        let defLine = uint32 (sig_.defLine - 1)   // IR lines are 1-based, LSP is 0-based
        let defCol  = uint32 sig_.defCol
        Some {
            Uri   = uri
            Range = { Start = { Line = defLine; Character = defCol }
                      End   = { Line = defLine; Character = defCol + uint32 word.Length } }
        }
    | _ ->

    // 2. Check externalFunctions (workspace cross-file)
    match externalFunctions.TryGetValue(word) with
    | true, extSig ->
        let extUri = "file://" + extSig.sourcePath
        Some {
            Uri   = extUri
            Range = { Start = { Line = 0u; Character = 0u }
                      End   = { Line = 0u; Character = 0u } }
        }
    | _ -> None
