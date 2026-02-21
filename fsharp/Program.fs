module Program

open System
open System.IO

// Escape a string value for JSON output.
let private jsonEscapeString (s: string) : string =
    let sb = System.Text.StringBuilder()
    for c in s do
        match c with
        | '"'  -> sb.Append("\\\"") |> ignore
        | '\\' -> sb.Append("\\\\") |> ignore
        | '\n' -> sb.Append("\\n")  |> ignore
        | '\r' -> sb.Append("\\r")  |> ignore
        | '\t' -> sb.Append("\\t")  |> ignore
        | c when int c < 32 ->
            sb.Append(sprintf "\\u%04x" (int c)) |> ignore
        | c    -> sb.Append(c) |> ignore
    sb.ToString()

// Serialize a single Token to a JSON object line.
let private tokenToJson (tok: Lexer.Token) : string =
    sprintf """{"kind": "%s", "value": "%s", "line": %d, "col": %d, "pos": %d}"""
        (jsonEscapeString tok.kind)
        (jsonEscapeString tok.value)
        tok.line
        tok.col
        tok.pos

[<EntryPoint>]
let main argv =
    if argv.Length < 1 then
        eprintfn "Usage: conformal-parse <file.m>"
        1
    else
        let filePath = argv.[0]
        if not (File.Exists filePath) then
            eprintfn "File not found: %s" filePath
            1
        else
            try
                let src = File.ReadAllText(filePath)
                let tokens = Lexer.lex src
                // Emit JSON array of tokens to stdout.
                printfn "["
                let last = tokens.Length - 1
                tokens |> List.iteri (fun i tok ->
                    let comma = if i < last then "," else ""
                    printfn "  %s%s" (tokenToJson tok) comma
                )
                printfn "]"
                0
            with
            | Lexer.LexError msg ->
                eprintfn "LexError: %s" msg
                2
            | ex ->
                eprintfn "Error: %s" ex.Message
                3
