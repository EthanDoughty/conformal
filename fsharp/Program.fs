module Program

open System
open System.IO

[<EntryPoint>]
let main argv =
    if argv.Length < 1 then
        Console.Error.WriteLine("Usage: conformal-parse <file.m>")
        1
    else
        let filePath = argv.[0]
        if not (File.Exists filePath) then
            Console.Error.WriteLine("File not found: " + filePath)
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
                Console.Error.WriteLine("ParseError: " + msg)
                2
            | Lexer.LexError msg ->
                Console.Error.WriteLine("LexError: " + msg)
                2
            | ex ->
                Console.Error.WriteLine("Error: " + ex.Message)
                3
