module PropertyTests

open FsCheck
open FsCheck.FSharp

// ============================================================================
// Generators
// ============================================================================

let private varNames = [| "n"; "m"; "k"; "p" |]
let private fieldNames = [| "x"; "y"; "z"; "w" |]
let private envVarNames = [| "A"; "B"; "x"; "y"; "z" |]

// -- SymDim generator ---------------------------------------------------------

let private genSymDim : Gen<SymDim.SymDim> =
    gen {
        let! numTerms = Gen.choose(0, 2)
        let! varIndices = Gen.listOfLength numTerms (Gen.choose(0, varNames.Length - 1))
        let! coeffs = Gen.listOfLength numTerms (Gen.choose(-5, 5))
        let mutable result = SymDim.SymDim.zero()
        for i in 0..numTerms-1 do
            let c = coeffs.[i]
            if c <> 0 then
                let coeff = { SymDim._terms = [([], SymDim.Rational.FromInt c)] }
                let v = SymDim.SymDim.var varNames.[varIndices.[i]]
                let term = SymDim.SymDim.mul coeff v
                result <- SymDim.SymDim.add result term
        return result
    }

// -- Dim generator (normalizes constant SymDims to Concrete) ------------------

let private genDim : Gen<Shapes.Dim> =
    Gen.frequency [
        (4, Gen.choose(0, 10) |> Gen.map Shapes.Concrete)
        (3, genSymDim |> Gen.map (fun s ->
            match SymDim.SymDim.constValue s with
            | Some cv -> Shapes.Concrete cv
            | None -> Shapes.Symbolic s))
        (2, Gen.constant Shapes.Unknown)
    ]

// -- Shape generator (depth-bounded) ------------------------------------------

let rec private genShapeD (depth: int) : Gen<Shapes.Shape> =
    let baseGens = [
        Gen.constant Shapes.Scalar
        gen { let! r = genDim in let! c = genDim in return Shapes.Matrix(r, c) }
        Gen.constant Shapes.StringShape
        Gen.constant Shapes.UnknownShape
        Gen.constant Shapes.Bottom
    ]
    if depth <= 0 then
        Gen.oneof baseGens
    else
        let structGen =
            gen {
                let! n = Gen.choose(0, 2)
                let! indices = Gen.listOfLength n (Gen.choose(0, fieldNames.Length - 1))
                let names = indices |> List.map (fun i -> fieldNames.[i]) |> List.distinct
                let! shapes = Gen.listOfLength names.Length (genShapeD (depth - 1))
                let fields = List.zip names shapes |> List.sortBy fst
                let! isOpen = Gen.elements [true; false]
                return Shapes.Struct(fields, isOpen)
            }
        let cellGen =
            gen { let! r = genDim in let! c = genDim in return Shapes.Cell(r, c, None) }
        let fhGen = Gen.constant (Shapes.FunctionHandle None)
        Gen.oneof (baseGens @ [structGen; cellGen; fhGen])

let private genShape = genShapeD 2

// -- Interval generators ------------------------------------------------------

let private genConcreteInterval : Gen<SharedTypes.Interval> =
    gen {
        let! a = Gen.choose(-20, 20)
        let! b = Gen.choose(-20, 20)
        return { lo = SharedTypes.Finite(min a b); hi = SharedTypes.Finite(max a b) }
    }

let private genIntervalOpt : Gen<SharedTypes.Interval option> =
    Gen.frequency [
        (4, genConcreteInterval |> Gen.map Some)
        (1, Gen.constant (Some { lo = SharedTypes.Unbounded; hi = SharedTypes.Unbounded }))
        (1, Gen.constant None)
    ]

// -- Env generator ------------------------------------------------------------

let private genEnv : Gen<Env.Env> =
    gen {
        let! n = Gen.choose(0, 4)
        let! indices = Gen.listOfLength n (Gen.choose(0, envVarNames.Length - 1))
        let names = indices |> List.map (fun i -> envVarNames.[i]) |> List.distinct
        let! shapes = Gen.listOfLength names.Length genShape
        let env = Env.Env.create()
        for i in 0..names.Length-1 do
            Env.Env.set env names.[i] shapes.[i]
        return env
    }

// -- IR generators (for no-crash) ---------------------------------------------

let private irOps = [| "+"; "-"; ".*"; "./" |]

let rec private genExprD (depth: int) : Gen<Ir.Expr> =
    let baseGens = [
        Gen.choose(-10, 10) |> Gen.map (fun n -> Ir.Const(1, 0, float n))
        Gen.choose(0, envVarNames.Length - 1) |> Gen.map (fun i -> Ir.Var(1, 0, envVarNames.[i]))
    ]
    if depth <= 0 then
        Gen.oneof baseGens
    else
        let binopGen =
            gen {
                let! opIdx = Gen.choose(0, irOps.Length - 1)
                let! l = genExprD (depth - 1)
                let! r = genExprD (depth - 1)
                return Ir.BinOp(1, 0, irOps.[opIdx], l, r)
            }
        Gen.oneof (baseGens @ [binopGen])

let private genExpr = genExprD 2

let private genProgram : Gen<Ir.Program> =
    gen {
        let! n = Gen.choose(1, 5)
        let! stmts = Gen.listOfLength n (gen {
            let! nameIdx = Gen.choose(0, envVarNames.Length - 1)
            let! expr = genExpr
            return Ir.Assign(1, 0, envVarNames.[nameIdx], expr)
        })
        return { Ir.body = stmts }
    }

// ============================================================================
// Arbitraries
// ============================================================================

let private arbShape = Arb.fromGen genShape
let private arbDim = Arb.fromGen genDim
let private arbSymDim = Arb.fromGen genSymDim
let private arbInterval = Arb.fromGen genConcreteInterval
let private arbIntervalOpt = Arb.fromGen genIntervalOpt
let private arbEnv = Arb.fromGen genEnv
let private arbProgram = Arb.fromGen genProgram

let private arbShapePair = Arb.fromGen (Gen.two genShape)
let private arbShapeTriple = Arb.fromGen (Gen.three genShape)
let private arbDimPair = Arb.fromGen (Gen.two genDim)
let private arbSymDimPair = Arb.fromGen (Gen.two genSymDim)
let private arbSymDimTriple = Arb.fromGen (Gen.three genSymDim)
let private arbIntervalPair = Arb.fromGen (Gen.two genConcreteInterval)
let private arbIntervalOptPair = Arb.fromGen (Gen.two genIntervalOpt)
let private arbEnvPair = Arb.fromGen (Gen.two genEnv)

// ============================================================================
// Runner
// ============================================================================

let private checkProp (name: string) (prop: Lazy<unit>) : bool =
    try
        prop.Force()
        printfn "  PASS  %s" name
        true
    with ex ->
        eprintfn "  FAIL  %s: %s" name (ex.Message.Split('\n').[0])
        false

let runPropertyTests () : int =
    printfn "=== Property-based tests ==="
    let results = [

        // -- A. Shape Lattice (8) --

        checkProp "A1 joinShape commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShapePair (fun (a, b) ->
                Shapes.joinShape a b = Shapes.joinShape b a)))

        checkProp "A2 joinShape idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.joinShape a a = a)))

        checkProp "A3 joinShape associative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShapeTriple (fun (a, b, c) ->
                let lhs = Shapes.joinShape (Shapes.joinShape a b) c
                let rhs = Shapes.joinShape a (Shapes.joinShape b c)
                lhs = rhs)))

        checkProp "A4 joinShape Bottom identity" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.joinShape Shapes.Bottom a = a
                && Shapes.joinShape a Shapes.Bottom = a)))

        checkProp "A5 joinShape Unknown absorbing" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.joinShape Shapes.UnknownShape a = Shapes.UnknownShape
                && Shapes.joinShape a Shapes.UnknownShape = Shapes.UnknownShape)))

        checkProp "A6 widenShape idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.widenShape a a = a)))

        checkProp "A7 widenShape Bottom identity" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.widenShape Shapes.Bottom a = a)))

        checkProp "A8 widenShape Unknown absorbing" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbShape (fun a ->
                Shapes.widenShape Shapes.UnknownShape a = Shapes.UnknownShape
                && Shapes.widenShape a Shapes.UnknownShape = Shapes.UnknownShape)))

        // -- B. Dim Lattice (5) --

        checkProp "B1 joinDim commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbDimPair (fun (a, b) ->
                Shapes.joinDim a b = Shapes.joinDim b a)))

        checkProp "B2 joinDim idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbDim (fun a ->
                Shapes.joinDim a a = a)))

        checkProp "B3 addDim commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbDimPair (fun (a, b) ->
                Shapes.addDim a b = Shapes.addDim b a)))

        checkProp "B4 mulDim commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbDimPair (fun (a, b) ->
                Shapes.mulDim a b = Shapes.mulDim b a)))

        checkProp "B5 addDim zero identity" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbDim (fun a ->
                Shapes.addDim (Shapes.Concrete 0) a = a
                && Shapes.addDim a (Shapes.Concrete 0) = a)))

        // -- C. SymDim Algebra (7) --

        checkProp "C1 SymDim add commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDimPair (fun (a, b) ->
                SymDim.SymDim.add a b = SymDim.SymDim.add b a)))

        checkProp "C2 SymDim add associative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDimTriple (fun (a, b, c) ->
                let lhs = SymDim.SymDim.add (SymDim.SymDim.add a b) c
                let rhs = SymDim.SymDim.add a (SymDim.SymDim.add b c)
                lhs = rhs)))

        checkProp "C3 SymDim add zero identity" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDim (fun a ->
                SymDim.SymDim.add a (SymDim.SymDim.zero()) = a
                && SymDim.SymDim.add (SymDim.SymDim.zero()) a = a)))

        checkProp "C4 SymDim add neg inverse" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDim (fun a ->
                SymDim.SymDim.add a (SymDim.SymDim.neg a) = SymDim.SymDim.zero())))

        checkProp "C5 SymDim mul commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDimPair (fun (a, b) ->
                SymDim.SymDim.mul a b = SymDim.SymDim.mul b a)))

        checkProp "C6 SymDim mul distributive" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbSymDimTriple (fun (a, b, c) ->
                let lhs = SymDim.SymDim.mul a (SymDim.SymDim.add b c)
                let rhs = SymDim.SymDim.add (SymDim.SymDim.mul a b) (SymDim.SymDim.mul a c)
                lhs = rhs)))

        checkProp "C7 SymDim evaluate consistent with substitute" (lazy
            let arbWithBindings = Arb.fromGen (gen {
                let! s = genSymDim
                let vars = SymDim.SymDim.variables s |> Set.toList
                let! vals = Gen.listOfLength vars.Length (Gen.choose(-10, 10))
                let bindings = List.zip vars vals |> Map.ofList
                return (s, bindings)
            })
            Check.QuickThrowOnFailure(Prop.forAll arbWithBindings (fun (s, bindings) ->
                let lhs = SymDim.SymDim.evaluate bindings s
                let rhs = SymDim.SymDim.constValue (SymDim.SymDim.substitute bindings s)
                lhs = rhs)))

        // -- D. Interval Lattice (5) --

        checkProp "D1 joinInterval commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbIntervalOptPair (fun (a, b) ->
                Intervals.joinInterval a b = Intervals.joinInterval b a)))

        checkProp "D2 joinInterval idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbIntervalOpt (fun a ->
                Intervals.joinInterval a a = a)))

        checkProp "D3 meetInterval commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbIntervalPair (fun (a, b) ->
                Intervals.meetInterval a b = Intervals.meetInterval b a)))

        checkProp "D4 meetInterval idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbInterval (fun a ->
                Intervals.meetInterval a a = Some a)))

        checkProp "D5 intervalAdd correct for concrete" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbIntervalPair (fun (a, b) ->
                match a.lo, a.hi, b.lo, b.hi with
                | SharedTypes.Finite la, SharedTypes.Finite ha,
                  SharedTypes.Finite lb, SharedTypes.Finite hb ->
                    let expected : SharedTypes.Interval option =
                        Some { lo = SharedTypes.Finite(la + lb)
                               hi = SharedTypes.Finite(ha + hb) }
                    Intervals.intervalAdd (Some a) (Some b) = expected
                | _ -> true)))

        // -- E. Env (2) --

        checkProp "E1 joinEnv commutative" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbEnvPair (fun (a, b) ->
                let j1 = Env.joinEnv a b
                let j2 = Env.joinEnv b a
                j1.bindings = j2.bindings)))

        checkProp "E2 joinEnv idempotent" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbEnv (fun a ->
                let j = Env.joinEnv a a
                j.bindings = a.bindings)))

        // -- F. No-Crash (1) --

        checkProp "F1 analyze random IR no crash" (lazy
            Check.QuickThrowOnFailure(Prop.forAll arbProgram (fun prog ->
                let ctx = Context.AnalysisContext()
                let (_env, _warnings) = Analysis.analyzeProgramIr prog ctx
                true)))
    ]

    let passed = results |> List.filter id |> List.length
    let total = results.Length
    printfn "=== Property tests: %d/%d passed ===" passed total
    if passed = total then 0 else 1
