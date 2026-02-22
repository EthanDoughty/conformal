module LspSymbols

open Ionide.LanguageServerProtocol.Types
open Ir

// ---------------------------------------------------------------------------
// getDocumentSymbols: document symbol provider for outline view.
// Port of lsp/symbols.py get_document_symbols().
// ---------------------------------------------------------------------------

/// getDocumentSymbols: walk IR program body for FunctionDef nodes.
/// Returns DocumentSymbol array (one per function definition).
/// sourceLines is used to compute end-of-line character positions.
let getDocumentSymbols
    (program: Program)
    (sourceLines: string array)
    : DocumentSymbol array =

    let symbols = System.Collections.Generic.List<DocumentSymbol>()

    for stmt in program.body do
        match stmt with
        | FunctionDef(line, _, name, parms, outputVars, body) ->

            // Detail string: (params) -> [outputs]
            let paramsStr  = parms      |> String.concat ", "
            let outputsStr = outputVars |> String.concat ", "
            let detail = "(" + paramsStr + ") -> [" + outputsStr + "]"

            // 1-based IR line -> 0-based LSP line
            let startLine = line - 1

            // End line: max statement line in body, or just the start line
            let endLine =
                if body.IsEmpty then startLine
                else
                    let maxBodyLine = body |> List.map (fun s -> s.Line) |> List.max
                    maxBodyLine - 1

            // Clamp to valid range
            let safeEndLine = max startLine endLine

            let endLineLen =
                if safeEndLine >= 0 && safeEndLine < sourceLines.Length then
                    sourceLines.[safeEndLine].Length
                else 0

            let startLineLen =
                if startLine >= 0 && startLine < sourceLines.Length then
                    sourceLines.[startLine].Length
                else 0

            let fullRange : Range = {
                Start = { Line = uint32 startLine; Character = 0u }
                End   = { Line = uint32 safeEndLine; Character = uint32 endLineLen }
            }

            let selRange : Range = {
                Start = { Line = uint32 startLine; Character = 0u }
                End   = { Line = uint32 startLine; Character = uint32 startLineLen }
            }

            let sym : DocumentSymbol = {
                Name           = name
                Kind           = SymbolKind.Function
                Range          = fullRange
                SelectionRange = selRange
                Detail         = Some detail
                Children       = None
                Tags           = None
                Deprecated     = None
            }

            symbols.Add(sym)

        | _ -> ()

    symbols.ToArray()
