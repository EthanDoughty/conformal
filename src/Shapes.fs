module Shapes

open SymDim

// ---------------------------------------------------------------------------
// Dim: abstract dimension (concrete int, symbolic SymDim, or unknown)
// ---------------------------------------------------------------------------

type Dim =
    | Concrete of int
    | Symbolic of SymDim
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
// Dimension helpers
// ---------------------------------------------------------------------------

let private toSymDim (d: Dim) : SymDim =
    match d with
    | Concrete n -> SymDim.const' n
    | Symbolic s -> s
    | Unknown    -> failwith "Cannot convert Unknown Dim to SymDim"

// Format a Dim for display in shape strings.
// Mirrors Python's _dim_str:
//   None (Unknown) -> "None"
//   int (Concrete) -> "3"
//   SymDim constant -> display as int
//   SymDim bare variable (single term, coeff 1, single var exp 1) -> bare name
//   SymDim expression -> "(expr)"
let dimStr (d: Dim) : string =
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

// ---------------------------------------------------------------------------
// Shape -> string (must be character-identical to Python __str__)
// ---------------------------------------------------------------------------

let rec shapeToString (s: Shape) : string =
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

// join_dim: same -> same; anything with Unknown -> Unknown; different -> Unknown
let joinDim (a: Dim) (b: Dim) : Dim =
    if a = b then a
    else Unknown

// widen_dim: same -> same; different -> Unknown
let widenDim (old: Dim) (newDim: Dim) : Dim =
    if old = newDim then old
    else Unknown

// dims_definitely_conflict: both known and provably different
let dimsDefinitelyConflict (a: Dim) (b: Dim) : bool =
    match a, b with
    | Unknown, _ | _, Unknown -> false
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

// add_dim: symbolic addition (None propagates)
let addDim (a: Dim) (b: Dim) : Dim =
    match a, b with
    | Unknown, _ | _, Unknown -> Unknown
    | Concrete ia, Concrete ib -> Concrete (ia + ib)
    | _ ->
        let sa = toSymDim a
        let sb = toSymDim b
        let result = SymDim.add sa sb
        match SymDim.constValue result with
        | Some cv -> Concrete cv
        | None    -> Symbolic result

// mul_dim: symbolic multiplication with short-circuits
let mulDim (a: Dim) (b: Dim) : Dim =
    // Short-circuit: 0 * x = 0
    if a = Concrete 0 || b = Concrete 0 then Concrete 0
    // Short-circuit: 1 * x = x
    elif a = Concrete 1 then b
    elif b = Concrete 1 then a
    else
        match a, b with
        | Unknown, _ | _, Unknown -> Unknown
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
