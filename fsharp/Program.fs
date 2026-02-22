module Program

open System
open System.IO

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
                let program = Parser.parseMATLAB src
                let json = Json.programToJson program
                Console.WriteLine(json)
                0
            with
            | Parser.ParseError msg ->
                eprintfn "ParseError: %s" msg
                2
            | Lexer.LexError msg ->
                eprintfn "LexError: %s" msg
                2
            | ex ->
                eprintfn "Error: %s" ex.Message
                3
