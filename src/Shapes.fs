module Shapes

open SymDim

// ---------------------------------------------------------------------------
// DimBound: flat constituent of Range (no nesting)
// BConcrete: exact integer bound
// BSymbolic: symbolic bound
// BUnknown: represents +infinity (hi) or 0 (lo), i.e. "don't know"
// ---------------------------------------------------------------------------

type DimBound =
    | BConcrete of int
    | BSymbolic of SymDim
    | BUnknown

// ---------------------------------------------------------------------------
// Dim: abstract dimension (concrete int, symbolic SymDim, range interval, or unknown)
// ---------------------------------------------------------------------------

type Dim =
    | Concrete of int
    | Symbolic of SymDim
    | Range    of lo: DimBound * hi: DimBound   // lo <= hi semantically
    | Unknown

// ---------------------------------------------------------------------------
// Shape: abstract shape domain (discriminated union)
// ---------------------------------------------------------------------------

type Shape =
    | Scalar
    | Matrix    of rows: Dim * cols: Dim
    | StringShape
    | Struct    of fields: (string * Shape) list * isOpen: bool
    | FunctionHandle of lambdaIds: Set<int> option
    | Cell      of rows: Dim * cols: Dim * elements: Map<int, Shape> option
    | UnknownShape
    | Bottom

// ---------------------------------------------------------------------------
// DimBound helpers
// ---------------------------------------------------------------------------

/// minBound: minimum of two DimBounds (for lo of hull).
/// BUnknown as lo represents 0 (minimum possible dim).
let private minBound (a: DimBound) (b: DimBound) : DimBound =
    match a, b with
    | BConcrete x, BConcrete y -> BConcrete (min x y)
    | BUnknown, _ -> BUnknown   // BUnknown as lo = 0 (the smallest possible)
    | _, BUnknown -> BUnknown
    | _ -> BUnknown  // can't compare symbolic: conservative

/// maxBound: maximum of two DimBounds (for hi of hull).
/// BUnknown as hi represents +infinity.
let private maxBound (a: DimBound) (b: DimBound) : DimBound =
    match a, b with
    | BConcrete x, BConcrete y -> BConcrete (max x y)
    | BUnknown, _ -> BUnknown   // BUnknown as hi = +infinity
    | _, BUnknown -> BUnknown
    | _ -> BUnknown  // can't compare symbolic: conservative

/// addBound: add two DimBounds (for interval arithmetic).
let addBound (a: DimBound) (b: DimBound) : DimBound =
    match a, b with
    | BConcrete x, BConcrete y -> BConcrete (x + y)
    | BUnknown, _ | _, BUnknown -> BUnknown
    | BSymbolic s, BConcrete n -> BSymbolic (SymDim.add s (SymDim.const' n))
    | BConcrete n, BSymbolic s -> BSymbolic (SymDim.add (SymDim.const' n) s)
    | BSymbolic a, BSymbolic b -> BSymbolic (SymDim.add a b)

/// dimToBounds: extract (lo, hi) DimBound pair from any Dim.
let dimToBounds (d: Dim) : DimBound * DimBound =
    match d with
    | Concrete n  -> (BConcrete n, BConcrete n)
    | Symbolic s  -> (BSymbolic s, BSymbolic s)
    | Range(lo, hi) -> (lo, hi)
    | Unknown     -> (BUnknown, BUnknown)

/// canonicalizeDim: normalize degenerate Range cases.
let canonicalizeDim (d: Dim) : Dim =
    match d with
    | Range(BConcrete a, BConcrete b) when a = b  -> Concrete a
    | Range(BSymbolic a, BSymbolic b) when a = b  -> Symbolic a
    | Range(BUnknown, BUnknown)                    -> Unknown
    | Range(BConcrete a, BConcrete b) when a > b  -> Unknown   // empty interval
    | Range(BConcrete a, _) when a < 0            ->
        // Clamp lo to 0 (MATLAB dims are non-negative)
        Range(BConcrete 0, snd (dimToBounds d))
    | _ -> d

/// boundsDisjoint: check if intervals [alo, ahi] and [blo, bhi] are provably non-overlapping.
/// Only provable when both sides have concrete bounds.
let private boundsDisjoint (alo: DimBound) (ahi: DimBound) (blo: DimBound) (bhi: DimBound) : bool =
    match alo, ahi, blo, bhi with
    | BConcrete al, BConcrete ah, BConcrete bl, BConcrete bh ->
        ah < bl || bh < al   // disjoint: a ends before b starts, or vice versa
    | _ -> false   // symbolic/unknown: can't prove disjoint

/// extendRange: extend range [lo, hi] to include a new point bound.
let private extendRange (lo: DimBound) (hi: DimBound) (pt: DimBound) : Dim =
    canonicalizeDim (Range(minBound lo pt, maxBound hi pt))

// ---------------------------------------------------------------------------
// Dimension helpers
// ---------------------------------------------------------------------------

let private toSymDim (d: Dim) : SymDim =
    match d with
    | Concrete n -> SymDim.const' n
    | Symbolic s -> s
    | Range _    -> failwith "Cannot convert Range Dim to SymDim"
    | Unknown    -> failwith "Cannot convert Unknown Dim to SymDim"

// Format a Dim for display in shape strings.
// Mirrors Python's _dim_str:
//   None (Unknown) -> "None"
//   int (Concrete) -> "3"
//   SymDim constant -> display as int
//   SymDim bare variable (single term, coeff 1, single var exp 1) -> bare name
//   SymDim expression -> "(expr)"
//   Range -> "lo..hi" with BUnknown displayed as empty string on that side
let private dimStrCache = System.Collections.Generic.Dictionary<Dim, string>()

let private dimStrCore (d: Dim) : string =
    match d with
    | Unknown    -> "None"
    | Concrete n -> string n
    | Symbolic s ->
        match SymDim.constValue s with
        | Some cv -> string cv
        | None ->
            // Check for bare variable: single term, coeff 1, single var with exp 1
            match s._terms with
            | [([_varName, 1], coeff)] when coeff = Rational.One ->
                fst (List.head (fst s._terms.[0]))
            | _ ->
                "(" + SymDim.toString s + ")"
    | Range(lo, hi) ->
        let loStr =
            match lo with
            | BConcrete n -> string n
            | BSymbolic s -> "(" + SymDim.toString s + ")"
            | BUnknown    -> ""
        let hiStr =
            match hi with
            | BConcrete n -> string n
            | BSymbolic s -> "(" + SymDim.toString s + ")"
            | BUnknown    -> ""
        loStr + ".." + hiStr

let dimStr (d: Dim) : string =
    match dimStrCache.TryGetValue(d) with
    | true, cached -> cached
    | _ ->
        let result = dimStrCore d
        dimStrCache.[d] <- result
        result

// ---------------------------------------------------------------------------
// Shape -> string (must be character-identical to Python __str__)
// ---------------------------------------------------------------------------

let private shapeStringCache = System.Collections.Generic.Dictionary<Shape, string>()

let rec shapeToString (s: Shape) : string =
    match shapeStringCache.TryGetValue(s) with
    | true, cached -> cached
    | _ ->
        let result = shapeToStringCore s
        shapeStringCache.[s] <- result
        result

and private shapeToStringCore (s: Shape) : string =
    match s with
    | Scalar           -> "scalar"
    | Matrix(r, c)     -> "matrix[" + dimStr r + " x " + dimStr c + "]"
    | StringShape      -> "string"
    | Struct(fields, isOpen) ->
        // Filter out bottom fields (internal-only)
        let fieldStrs =
            fields
            |> List.choose (fun (name, shape) ->
                if shape = Bottom then None
                else Some (name + ": " + shapeToString shape))
        let allParts =
            if isOpen then fieldStrs @ ["..."] else fieldStrs
        "struct{" + String.concat ", " allParts + "}"
    | FunctionHandle _ -> "function_handle"
    | Cell(r, c, _)   -> "cell[" + dimStr r + " x " + dimStr c + "]"
    | UnknownShape     -> "unknown"
    | Bottom           -> "bottom"

// ---------------------------------------------------------------------------
// Dimension lattice operations
// ---------------------------------------------------------------------------

// join_dim: same -> same; Range + anything -> convex hull; different non-Range -> Unknown
// IMPORTANT: joinDim is Range-PRESERVING but NOT Range-CREATING.
// Two non-Range values that differ still produce Unknown (unchanged behavior).
let joinDim (a: Dim) (b: Dim) : Dim =
    if a = b then a
    else
        match a, b with
        // Range-Range: convex hull
        | Range(alo, ahi), Range(blo, bhi) ->
            canonicalizeDim (Range(minBound alo blo, maxBound ahi bhi))
        // Range-Point or Point-Range: embed point, then hull
        | Range(lo, hi), other | other, Range(lo, hi) ->
            let (olo, ohi) = dimToBounds other
            canonicalizeDim (Range(minBound lo olo, maxBound hi ohi))
        // Two non-Range values that differ: Unknown (NOT Range — unchanged behavior)
        | _ -> Unknown

// widen_dim: same -> same; different -> Unknown (Phase 1: no Range creation yet)
// Phase 3 will change this to produce Ranges.
let widenDim (old: Dim) (newDim: Dim) : Dim =
    if old = newDim then old
    else
        match old, newDim with
        // Point -> Point: create Range (Phase 3 activation)
        | Concrete a, Concrete b -> canonicalizeDim (Range(BConcrete (min a b), BConcrete (max a b)))
        | Concrete a, Symbolic s -> Range(BConcrete a, BSymbolic s)  // assume symbolic >= concrete
        | Symbolic s, Concrete a -> Range(BConcrete a, BSymbolic s)
        // Point -> Unknown: open-ended range
        | Concrete a, Unknown -> Range(BConcrete a, BUnknown)
        | Symbolic s, Unknown -> Range(BSymbolic s, BUnknown)
        // Range -> Point: extend bounds
        | Range(lo, hi), Concrete n -> extendRange lo hi (BConcrete n)
        | Range(lo, hi), Symbolic s -> extendRange lo hi (BSymbolic s)
        // Range -> Unknown: open-ended
        | Range(lo, _), Unknown -> canonicalizeDim (Range(lo, BUnknown))
        // Range -> Range: hull
        | Range(lo1, hi1), Range(lo2, hi2) -> canonicalizeDim (Range(minBound lo1 lo2, maxBound hi1 hi2))
        // Point -> Range: incorporate point into the range (post-loop join: preLoop is point, stabilized is Range)
        | Concrete a, Range(lo, hi) -> extendRange lo hi (BConcrete a)
        | Symbolic s, Range(lo, hi) -> extendRange lo hi (BSymbolic s)
        // Unknown old -> Unknown (can't narrow after going to top)
        | Unknown, _ -> Unknown
        // Symbolic -> Symbolic (different): conservative
        | _ -> Unknown

// dims_definitely_conflict: both known and provably different (or disjoint ranges)
let dimsDefinitelyConflict (a: Dim) (b: Dim) : bool =
    match a, b with
    | Unknown, _ | _, Unknown -> false
    | Range _, _ | _, Range _ ->
        // Range vs anything: check if intervals are provably disjoint
        let (alo, ahi) = dimToBounds a
        let (blo, bhi) = dimToBounds b
        boundsDisjoint alo ahi blo bhi
    | _ when a = b -> false
    | Concrete ia, Concrete ib -> ia <> ib
    | _ ->
        // Check if difference is a nonzero constant
        try
            let sa = toSymDim a
            let sb = toSymDim b
            let diff = SymDim.sub sa sb
            match SymDim.constValue diff with
            | Some cv when cv <> 0 -> true
            | _ -> false
        with _ -> false

// add_dim: symbolic addition (Unknown propagates); Range gets interval arithmetic
let addDim (a: Dim) (b: Dim) : Dim =
    match a, b with
    | Unknown, _ | _, Unknown -> Unknown
    | Range(alo, ahi), _ ->
        let blo, bhi = dimToBounds b
        canonicalizeDim (Range(addBound alo blo, addBound ahi bhi))
    | _, Range(blo, bhi) ->
        let alo, ahi = dimToBounds a
        canonicalizeDim (Range(addBound alo blo, addBound ahi bhi))
    | Concrete ia, Concrete ib -> Concrete (ia + ib)
    | _ ->
        let sa = toSymDim a
        let sb = toSymDim b
        let result = SymDim.add sa sb
        match SymDim.constValue result with
        | Some cv -> Concrete cv
        | None    -> Symbolic result

// mul_dim: symbolic multiplication with short-circuits; Range gets interval arithmetic
let mulDim (a: Dim) (b: Dim) : Dim =
    // Short-circuit: 0 * x = 0
    if a = Concrete 0 || b = Concrete 0 then Concrete 0
    // Short-circuit: 1 * x = x
    elif a = Concrete 1 then b
    elif b = Concrete 1 then a
    else
        match a, b with
        | Unknown, _ | _, Unknown -> Unknown
        | Range(alo, ahi), Range(blo, bhi) ->
            // 4-corner interval multiplication (non-negative dims: min of corners = lo*lo, max = hi*hi)
            match alo, ahi, blo, bhi with
            | BConcrete al, BConcrete ah, BConcrete bl, BConcrete bh ->
                let corners = [ al*bl; al*bh; ah*bl; ah*bh ]
                canonicalizeDim (Range(BConcrete (List.min corners), BConcrete (List.max corners)))
            | _ -> Unknown   // symbolic bounds in mul: conservative
        | Range(alo, ahi), Concrete n ->
            // Range * concrete scalar
            let blo, bhi = BConcrete n, BConcrete n
            match alo, ahi with
            | BConcrete al, BConcrete ah ->
                let corners = [ al*n; ah*n ]
                canonicalizeDim (Range(BConcrete (List.min corners), BConcrete (List.max corners)))
            | _ -> Unknown
        | Concrete n, Range(blo, bhi) ->
            match blo, bhi with
            | BConcrete bl, BConcrete bh ->
                let corners = [ n*bl; n*bh ]
                canonicalizeDim (Range(BConcrete (List.min corners), BConcrete (List.max corners)))
            | _ -> Unknown
        | Concrete ia, Concrete ib -> Concrete (ia * ib)
        | _ ->
            let sa = toSymDim a
            let sb = toSymDim b
            let result = SymDim.mul sa sb
            match SymDim.constValue result with
            | Some cv -> Concrete cv
            | None    -> Symbolic result

// sub_dim: a - b
let subDim (a: Dim) (b: Dim) : Dim =
    match b with
    | Unknown -> Unknown
    | _ -> addDim a (mulDim (Concrete -1) b)

// sum_dims: fold addDim over a list
let sumDims (dims: Dim list) : Dim =
    match dims with
    | []    -> Concrete 0
    | first :: rest -> List.fold addDim first rest

// ---------------------------------------------------------------------------
// Generic shape traversal (join_shape / widen_shape share the same structure)
// ---------------------------------------------------------------------------

let rec traverseShapes (dimOp: Dim -> Dim -> Dim) (s1: Shape) (s2: Shape) : Shape =
    match s1, s2 with
    // Bottom is identity
    | Bottom, _      -> s2
    | _,      Bottom -> s1
    // Unknown is absorbing top
    | UnknownShape, _ | _, UnknownShape -> UnknownShape
    // Same-kind arms
    | Scalar, Scalar -> Scalar
    | Matrix(r1, c1), Matrix(r2, c2) ->
        Matrix(dimOp r1 r2, dimOp c1 c2)
    | StringShape, StringShape -> StringShape
    | FunctionHandle ids1, FunctionHandle ids2 ->
        let mergedIds =
            match ids1, ids2 with
            | None, _ | _, None -> None
            | Some a, Some b    -> Some (Set.union a b)
        FunctionHandle mergedIds
    | Struct(fields1, open1), Struct(fields2, open2) ->
        let resultOpen = open1 || open2
        let dict1 = Map.ofList fields1
        let dict2 = Map.ofList fields2
        let allFields = Set.union (dict1 |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
                                  (dict2 |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
        let mergedFields =
            allFields
            |> Set.toList
            |> List.sortBy id
            |> List.map (fun fieldName ->
                let default1 = if open1 then UnknownShape else Bottom
                let default2 = if open2 then UnknownShape else Bottom
                let f1 = defaultArg (Map.tryFind fieldName dict1) default1
                let f2 = defaultArg (Map.tryFind fieldName dict2) default2
                (fieldName, traverseShapes dimOp f1 f2))
        Struct(mergedFields, resultOpen)
    | Cell(r1, c1, elems1), Cell(r2, c2, elems2) ->
        let mergedRows = dimOp r1 r2
        let mergedCols = dimOp c1 c2
        let mergedElems =
            match elems1, elems2 with
            | None, _ | _, None -> None
            | Some m1, Some m2 ->
                let allIdx = Set.union (m1 |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
                                       (m2 |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
                let elemDict =
                    allIdx
                    |> Set.toList
                    |> List.choose (fun idx ->
                        let e1 = defaultArg (Map.tryFind idx m1) Bottom
                        let e2 = defaultArg (Map.tryFind idx m2) Bottom
                        let merged = traverseShapes dimOp e1 e2
                        if merged = Bottom then None else Some (idx, merged))
                    |> Map.ofList
                if Map.isEmpty elemDict then None else Some elemDict
        Cell(mergedRows, mergedCols, mergedElems)
    // Different kinds -> Unknown
    | _ -> UnknownShape

let joinShape (s1: Shape) (s2: Shape) : Shape = traverseShapes joinDim s1 s2
let widenShape (old: Shape) (newShape: Shape) : Shape = traverseShapes widenDim old newShape

// ---------------------------------------------------------------------------
// Predicates (convenience)
// ---------------------------------------------------------------------------

let isScalar s         = s = Scalar
let isMatrix s         = match s with Matrix _ -> true | _ -> false
let isUnknown s        = s = UnknownShape
let isBottom s         = s = Bottom
let isString s         = s = StringShape
let isStruct s         = match s with Struct _ -> true | _ -> false
let isCell s           = match s with Cell _ -> true | _ -> false
let isFunctionHandle s = match s with FunctionHandle _ -> true | _ -> false
let isNumeric s        = match s with Scalar | Matrix _ | StringShape -> true | _ -> false
let isEmptyMatrix s    = match s with Matrix(Concrete 0, Concrete 0) -> true | _ -> false
