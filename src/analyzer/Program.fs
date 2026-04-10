// Conformal: Static Shape Analysis for MATLAB
// author: matrix[1 x 1] Ethan Doughty, 2026
//
// Executable entry point. Splits into two run modes: the stdio-based
// LSP server when --lsp is passed, and the regular CLI otherwise.

module Program

[<EntryPoint>]
let main argv =
    if argv |> Array.contains "--lsp" then
        LspServer.startLsp ()
    else
        Cli.run argv
