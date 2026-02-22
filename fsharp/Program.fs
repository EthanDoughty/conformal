module Program

open System
open System.IO
open Shapes
open SymDim

// ---------------------------------------------------------------------------
// Smoke test for Phase 1: SymDim + Shapes + Env
// ---------------------------------------------------------------------------

let runShapesTest () : int =
    let mutable failures = 0

    let check (label: string) (expected: string) (actual: string) =
        if expected = actual then
            Console.WriteLine("  PASS  " + label + " => " + actual)
        else
            Console.Error.WriteLine("  FAIL  " + label + ": expected \"" + expected + "\" got \"" + actual + "\"")
            failures <- failures + 1

    Console.WriteLine("=== Shape string tests ===")
    check "Scalar"          "scalar"          (shapeToString Scalar)
    check "Matrix 3x4"      "matrix[3 x 4]"   (shapeToString (Matrix(Concrete 3, Concrete 4)))
    check "Matrix n x None" "matrix[n x None]" (shapeToString (Matrix(Symbolic (SymDim.var "n"), Unknown)))
    check "Matrix None x 1" "matrix[None x 1]" (shapeToString (Matrix(Unknown, Concrete 1)))
    check "String"          "string"           (shapeToString StringShape)
    check "FunctionHandle"  "function_handle"  (shapeToString (FunctionHandle None))
    check "UnknownShape"    "unknown"          (shapeToString UnknownShape)
    check "Bottom"          "bottom"           (shapeToString Bottom)
    check "Struct empty"    "struct{}"         (shapeToString (Struct([], false)))
    check "Cell 2x1"        "cell[2 x 1]"      (shapeToString (Cell(Concrete 2, Concrete 1, None)))
    check "Matrix n x 1"    "matrix[n x 1]"    (shapeToString (Matrix(Symbolic (SymDim.var "n"), Concrete 1)))

    Console.WriteLine("=== SymDim arithmetic tests ===")
    let n = SymDim.var "n"
    let m = SymDim.var "m"
    let three = SymDim.const' 3

    check "var n"     "n"      (SymDim.toString n)
    check "const 3"   "3"      (SymDim.toString three)
    check "zero"      "0"      (SymDim.toString (SymDim.zero ()))
    check "n+3"       "n+3"    (SymDim.toString (SymDim.add n three))
    check "b-a"       "-a+b"   (SymDim.toString (SymDim.sub (SymDim.var "b") (SymDim.var "a")))
    check "n/2"       "n/2"    (SymDim.toString (SymDim.div n 2))
    check "n+m"       "m+n"    (SymDim.toString (SymDim.add n m))

    Console.WriteLine("=== Dim join/widen tests ===")
    let dimToStr d = match d with Concrete n -> "Concrete " + string n | Symbolic _ -> "Symbolic" | Unknown -> "Unknown"
    check "joinDim 3 3"     "Concrete 3" (dimToStr (joinDim (Concrete 3) (Concrete 3)))
    check "joinDim 3 4"     "Unknown"    (dimToStr (joinDim (Concrete 3) (Concrete 4)))
    check "widenDim 3 3"    "Concrete 3" (dimToStr (widenDim (Concrete 3) (Concrete 3)))
    check "widenDim 3 4"    "Unknown"    (dimToStr (widenDim (Concrete 3) (Concrete 4)))
    check "addDim 3 4"      "Concrete 7" (dimToStr (addDim (Concrete 3) (Concrete 4)))

    Console.WriteLine("=== Env tests ===")
    let env = Env.Env.create ()
    Env.Env.set env "x" Scalar
    Env.Env.set env "A" (Matrix(Concrete 3, Concrete 4))
    check "env.get x"      "scalar"         (shapeToString (Env.Env.get env "x"))
    check "env.get A"      "matrix[3 x 4]"  (shapeToString (Env.Env.get env "A"))
    check "env.get z"      "bottom"          (shapeToString (Env.Env.get env "z"))
    check "hasLocal x"     "True"            (string (Env.Env.hasLocal env "x"))
    check "contains x"     "True"            (string (Env.Env.contains env "x"))
    check "contains z"     "False"           (string (Env.Env.contains env "z"))

    let child = Env.Env.pushScope env
    Env.Env.set child "y" StringShape
    check "child.get y"    "string"          (shapeToString (Env.Env.get child "y"))
    check "child.get x"    "scalar"          (shapeToString (Env.Env.get child "x"))
    check "child.hasLocal x" "False"         (string (Env.Env.hasLocal child "x"))

    Console.WriteLine("=== dimsDefinitelyConflict tests ===")
    check "conflict 3 4"    "True"  (string (dimsDefinitelyConflict (Concrete 3) (Concrete 4)))
    check "no conflict 3 3" "False" (string (dimsDefinitelyConflict (Concrete 3) (Concrete 3)))
    check "no conflict unk" "False" (string (dimsDefinitelyConflict Unknown (Concrete 3)))

    Console.WriteLine("")
    if failures = 0 then
        Console.WriteLine("All shape smoke tests PASSED")
        0
    else
        Console.Error.WriteLine(string failures + " shape smoke test(s) FAILED")
        1

[<EntryPoint>]
let main argv =
    if argv.Length >= 1 && argv.[0] = "--test-shapes" then
        runShapesTest ()
    elif argv.Length < 1 then
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
