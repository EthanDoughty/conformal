module Program

open System
open System.IO

let private migrateFile (inputFile: string) (outputFile: string option) (toStdout: bool) : int =
    if not (File.Exists inputFile) then
        eprintfn "Error: file not found: %s" inputFile
        1
    else
        try
            let src = File.ReadAllText(inputFile)
            let (program, _) = Parser.parseMATLAB src

            // Run shape analysis
            let ctx = Context.AnalysisContext()
            let (env, _diags) = Analysis.analyzeProgramIr program ctx

            // Run copy semantics analysis
            let copySites = CopySemantics.findCopySites program.body ctx.shapeAnnotations

            // Create translation context
            let tctx : Translate.TranslateContext = {
                shapeAnnotations = ctx.shapeAnnotations
                copySites = copySites
                env = env
                usedImports = Set.empty
                currentReturnVars = []
                functionDepth = 0
            }

            // Translate
            let sourceFileName = Path.GetFileName(inputFile)
            let pyProgram = Translate.translateProgram program tctx sourceFileName

            // Emit
            let lines = Emit.emitProgram pyProgram
            let output = (lines |> String.concat "\n") + "\n"

            if toStdout || outputFile.IsNone then
                printf "%s" output
            else
                File.WriteAllText(outputFile.Value, output)
                eprintfn "Wrote %s" outputFile.Value
            0
        with
        | Parser.ParseError(msg, _, _, _, _) ->
            eprintfn "ParseError: %s" msg
            1
        | Lexer.LexError(msg, _, _, _, _) ->
            eprintfn "LexError: %s" msg
            1
        | ex ->
            eprintfn "Error: %s" ex.Message
            1

let private runMigrateTests () : int =
    // Find tests/migrate/ directory by searching upward
    let rec findUp (dir: string) =
        let candidate = Path.Combine(dir, "tests", "migrate")
        if Directory.Exists candidate then candidate
        else
            let parent = Path.GetDirectoryName(dir)
            if parent = null || parent = dir then candidate
            else findUp parent
    let testDir = findUp (Directory.GetCurrentDirectory())

    if not (Directory.Exists testDir) then
        eprintfn "Test directory not found: %s" testDir
        1
    else
        let mFiles = Directory.GetFiles(testDir, "*.m") |> Array.sort
        let mutable passed = 0
        let mutable failed = 0

        for mFile in mFiles do
            let expectedFile = Path.ChangeExtension(mFile, ".py.expected")
            if File.Exists expectedFile then
                let testName = Path.GetFileNameWithoutExtension(mFile)
                try
                    let src = File.ReadAllText(mFile)
                    let (program, _) = Parser.parseMATLAB src

                    let ctx = Context.AnalysisContext()
                    let (env, _diags) = Analysis.analyzeProgramIr program ctx

                    let copySites = CopySemantics.findCopySites program.body ctx.shapeAnnotations
                    let tctx : Translate.TranslateContext = {
                        shapeAnnotations = ctx.shapeAnnotations
                        copySites = copySites
                        env = env
                        usedImports = Set.empty
                        currentReturnVars = []
                        functionDepth = 0
                    }

                    let sourceFileName = Path.GetFileName(mFile)
                    let pyProgram = Translate.translateProgram program tctx sourceFileName
                    let lines = Emit.emitProgram pyProgram
                    let actual = (lines |> String.concat "\n").TrimEnd() + "\n"

                    let expected = (File.ReadAllText(expectedFile)).Replace("\r\n", "\n").TrimEnd() + "\n"

                    if actual = expected then
                        passed <- passed + 1
                    else
                        failed <- failed + 1
                        eprintfn "FAIL: %s" testName
                        // Show diff
                        let actualLines = actual.Split('\n')
                        let expectedLines = expected.Split('\n')
                        let maxLines = max actualLines.Length expectedLines.Length
                        for i in 0..maxLines-1 do
                            let a = if i < actualLines.Length then actualLines.[i] else "<missing>"
                            let e = if i < expectedLines.Length then expectedLines.[i] else "<missing>"
                            if a <> e then
                                eprintfn "  line %d:" (i + 1)
                                eprintfn "    expected: %s" e
                                eprintfn "    actual:   %s" a
                with ex ->
                    failed <- failed + 1
                    eprintfn "FAIL: %s - %s" (Path.GetFileNameWithoutExtension(mFile)) ex.Message

        let total = passed + failed
        if failed = 0 then
            printfn "===== Migrate tests: %d/%d passed =====" passed total
            0
        else
            eprintfn "===== Migrate tests: %d/%d passed, %d failed =====" passed total failed
            1

/// Usage: conformal-migrate <input.m> [-o output.py] [--stdout] [--test-migrate]
[<EntryPoint>]
let main argv =
    let args = Array.toList argv

    if List.contains "--test-migrate" args then
        runMigrateTests ()
    elif args.IsEmpty || List.contains "--help" args then
        eprintfn "Usage: conformal-migrate <input.m> [-o output.py] [--stdout]"
        eprintfn "       conformal-migrate --test-migrate"
        if args.IsEmpty then 1 else 0
    else
        let inputFile = args.[0]
        let outputFile =
            match args |> List.tryFindIndex (fun a -> a = "-o") with
            | Some i when i + 1 < args.Length -> Some args.[i + 1]
            | _ -> None
        let toStdout = List.contains "--stdout" args
        migrateFile inputFile outputFile toStdout
