module Program

[<EntryPoint>]
let main argv =
    if argv |> Array.contains "--lsp" then
        LspServer.startLsp ()
    else
        Cli.run argv
