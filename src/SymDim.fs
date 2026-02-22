module SymDim

open System.Collections.Generic

// ---------------------------------------------------------------------------
// Rational number (numerator/denominator as int64, always normalized)
// ---------------------------------------------------------------------------

[<Struct; CustomEquality; NoComparison>]
type Rational private (num: int64, den: int64) =
    member _.Numerator   = num
    member _.Denominator = den

    static member private Normalize(n: int64, d: int64) =
        if d = 0L then failwith "Rational: division by zero"
        let sign = if d < 0L then -1L else 1L
        let n' = n * sign
        let d' = d * sign
        let rec gcd a b = if b = 0L then a else gcd b (a % b)
        let g = gcd (abs n') d'
        Rational(n' / g, d' / g)

    static member Create(n: int64, d: int64) = Rational.Normalize(n, d)
    static member FromInt(n: int) = Rational.Normalize(int64 n, 1L)
    static member Zero = Rational.Normalize(0L, 1L)
    static member One  = Rational.Normalize(1L, 1L)

    static member (+)(a: Rational, b: Rational) =
        Rational.Normalize(a.Numerator * b.Denominator + b.Numerator * a.Denominator,
                           a.Denominator * b.Denominator)
    static member (-)(a: Rational, b: Rational) =
        Rational.Normalize(a.Numerator * b.Denominator - b.Numerator * a.Denominator,
                           a.Denominator * b.Denominator)
    static member (*)(a: Rational, b: Rational) =
        Rational.Normalize(a.Numerator * b.Numerator, a.Denominator * b.Denominator)
    static member (/)(a: Rational, b: Rational) =
        if b.Numerator = 0L then failwith "Rational: division by zero"
        Rational.Normalize(a.Numerator * b.Denominator, a.Denominator * b.Numerator)
    static member (~-)(a: Rational) = Rational.Normalize(-a.Numerator, a.Denominator)

    member this.IsZero    = this.Numerator = 0L
    member this.IsInteger = this.Denominator = 1L
    member this.ToInt64   = this.Numerator  // only valid when IsInteger

    override this.Equals(obj) =
        match obj with
        | :? Rational as r -> this.Numerator = r.Numerator && this.Denominator = r.Denominator
        | _ -> false
    override this.GetHashCode() = hash (this.Numerator, this.Denominator)

// ---------------------------------------------------------------------------
// Monomial: sorted list of (variable_name, exponent) pairs
// Empty list = constant monomial (1)
// ---------------------------------------------------------------------------

type Monomial = (string * int) list

// Sort key for monomials: higher degree first, then lexicographic
let private monoKey (mono: Monomial) : int * Monomial =
    let degree = mono |> List.sumBy snd
    (-degree, mono)

// ---------------------------------------------------------------------------
// SymDim: multivariate polynomial over rational coefficients
// Represented as a sorted association list: (Monomial * Rational) list
// Canonical: sorted by monoKey, no zero coefficients
// ---------------------------------------------------------------------------

type SymDim = { _terms: (Monomial * Rational) list }

// ---------------------------------------------------------------------------
// Module SymDim: constructors and operations
// ---------------------------------------------------------------------------

module SymDim =

    let zero () : SymDim = { _terms = [] }

    let const' (value: int) : SymDim =
        if value = 0 then zero ()
        else { _terms = [ ([], Rational.FromInt value) ] }

    let var (name: string) : SymDim =
        { _terms = [ ([name, 1], Rational.One) ] }

    // Collect terms from a dictionary into canonical sorted list
    let private canonicalize (termDict: Dictionary<Monomial, Rational>) : (Monomial * Rational) list =
        termDict
        |> Seq.choose (fun kv -> if kv.Value.IsZero then None else Some (kv.Key, kv.Value))
        |> Seq.toList
        |> List.sortBy (fun (mono, _) -> monoKey mono)

    let add (a: SymDim) (b: SymDim) : SymDim =
        let d = Dictionary<Monomial, Rational>(HashIdentity.Structural)
        for (mono, coeff) in a._terms do
            if d.ContainsKey(mono) then d.[mono] <- d.[mono] + coeff
            else d.[mono] <- coeff
        for (mono, coeff) in b._terms do
            if d.ContainsKey(mono) then d.[mono] <- d.[mono] + coeff
            else d.[mono] <- coeff
        { _terms = canonicalize d }

    let neg (a: SymDim) : SymDim =
        { _terms = a._terms |> List.map (fun (mono, coeff) -> (mono, -coeff)) }

    let sub (a: SymDim) (b: SymDim) : SymDim = add a (neg b)

    let mul (a: SymDim) (b: SymDim) : SymDim =
        let d = Dictionary<Monomial, Rational>(HashIdentity.Structural)
        for (mono1, coeff1) in a._terms do
            for (mono2, coeff2) in b._terms do
                let varDict = Dictionary<string, int>()
                for (v, e) in mono1 do
                    if varDict.ContainsKey(v) then varDict.[v] <- varDict.[v] + e
                    else varDict.[v] <- e
                for (v, e) in mono2 do
                    if varDict.ContainsKey(v) then varDict.[v] <- varDict.[v] + e
                    else varDict.[v] <- e
                let newMono =
                    varDict |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Seq.toList |> List.sortBy fst
                let newCoeff = coeff1 * coeff2
                if d.ContainsKey(newMono) then d.[newMono] <- d.[newMono] + newCoeff
                else d.[newMono] <- newCoeff
        { _terms = canonicalize d }

    let div (a: SymDim) (divisor: int) : SymDim =
        if divisor = 0 then failwith "SymDim.div: division by zero"
        let fracDiv = Rational.FromInt divisor
        { _terms = a._terms |> List.map (fun (mono, coeff) -> (mono, coeff / fracDiv)) }

    let divBySymDim (a: SymDim) (b: SymDim) : SymDim =
        match b._terms with
        | [([], c)] when c.IsInteger -> div a (int c.ToInt64)
        | _ -> failwith "SymDim: division only supported for constant divisors"

    let isConst (s: SymDim) : bool =
        s._terms.IsEmpty || (s._terms.Length = 1 && fst s._terms.[0] = [])

    let constValue (s: SymDim) : int option =
        match s._terms with
        | [] -> Some 0
        | [([], c)] when c.IsInteger -> Some (int c.ToInt64)
        | _ -> None

    let variables (s: SymDim) : Set<string> =
        s._terms
        |> List.collect (fun (mono, _) -> mono |> List.map fst)
        |> Set.ofList

    let substitute (bindings: Map<string, int>) (s: SymDim) : SymDim =
        let mutable result = zero ()
        for (mono, coeff) in s._terms do
            let mutable termValue =
                if coeff = Rational.One then const' 1
                else { _terms = [([], coeff)] }
            for (varName, exp) in mono do
                match Map.tryFind varName bindings with
                | Some concreteVal ->
                    let concrete = Rational.Create(int64 (pown concreteVal exp), 1L)
                    termValue <- { _terms = termValue._terms |> List.map (fun (m, c) -> (m, c * concrete)) }
                | None ->
                    let varFactor = var varName
                    let mutable factor = const' 1
                    for _ in 1..exp do
                        factor <- mul factor varFactor
                    termValue <- mul termValue factor
            result <- add result termValue
        result

    let evaluate (bindings: Map<string, int>) (s: SymDim) : int option =
        constValue (substitute bindings s)

    let private formatMono (mono: Monomial) : string =
        mono
        |> List.map (fun (v, e) -> if e = 1 then v else v + "^" + string e)
        |> String.concat "*"

    let toString (s: SymDim) : string =
        if s._terms.IsEmpty then "0"
        else
            let parts =
                s._terms |> List.map (fun (mono, coeff) ->
                    if mono = [] then
                        // Constant term
                        if coeff.IsInteger then string coeff.Numerator
                        else "(" + string coeff.Numerator + "/" + string coeff.Denominator + ")"
                    else
                        let varStr = formatMono mono
                        // Special case: single-var rational n/d where numerator=1 -> "n/d"
                        if not coeff.IsInteger && mono.Length = 1 && snd mono.[0] = 1 && coeff.Numerator = 1L then
                            varStr + "/" + string coeff.Denominator
                        elif coeff = Rational.One then
                            varStr
                        elif coeff = -Rational.One then
                            "-" + varStr
                        elif coeff.IsInteger then
                            string coeff.Numerator + "*" + varStr
                        else
                            "(" + string coeff.Numerator + "/" + string coeff.Denominator + ")*" + varStr
                )
            let mutable result = parts.[0]
            for part in parts |> List.tail do
                if part.StartsWith("-") then result <- result + part
                else result <- result + "+" + part
            result
