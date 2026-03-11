module BuiltinMap

open PyAst

type ArgTransform =
    | Direct
    | TupleFirstN of int
    | AttrStyle of string
    | AttrStyleDim
    | MaxShape
    | SizeAttr

type BuiltinMapping = {
    pythonFunc: string
    argTransform: ArgTransform
    needsOrderF: bool
}

let private builtinTable =
    dict [
        "zeros",     { pythonFunc = "np.zeros";          argTransform = TupleFirstN 2; needsOrderF = false }
        "ones",      { pythonFunc = "np.ones";           argTransform = TupleFirstN 2; needsOrderF = false }
        "eye",       { pythonFunc = "np.eye";            argTransform = Direct;        needsOrderF = false }
        "rand",      { pythonFunc = "np.random.rand";    argTransform = Direct;        needsOrderF = false }
        "randn",     { pythonFunc = "np.random.randn";   argTransform = Direct;        needsOrderF = false }
        "linspace",  { pythonFunc = "np.linspace";       argTransform = Direct;        needsOrderF = false }
        "reshape",   { pythonFunc = "np.reshape";        argTransform = TupleFirstN 2; needsOrderF = true  }
        "size",      { pythonFunc = ".shape";            argTransform = AttrStyleDim; needsOrderF = false }
        "length",    { pythonFunc = "max(.shape)";       argTransform = MaxShape;      needsOrderF = false }
        "numel",     { pythonFunc = ".size";             argTransform = SizeAttr;      needsOrderF = false }
        "inv",       { pythonFunc = "np.linalg.inv";     argTransform = Direct;        needsOrderF = false }
        "det",       { pythonFunc = "np.linalg.det";     argTransform = Direct;        needsOrderF = false }
        "eig",       { pythonFunc = "np.linalg.eig";     argTransform = Direct;        needsOrderF = false }
        "svd",       { pythonFunc = "np.linalg.svd";     argTransform = Direct;        needsOrderF = false }
        "pinv",      { pythonFunc = "np.linalg.pinv";    argTransform = Direct;        needsOrderF = false }
        "norm",      { pythonFunc = "np.linalg.norm";    argTransform = Direct;        needsOrderF = false }
        "diag",      { pythonFunc = "np.diag";           argTransform = Direct;        needsOrderF = false }
        "sum",       { pythonFunc = "np.sum";            argTransform = Direct;        needsOrderF = false }
        "max",       { pythonFunc = "np.max";            argTransform = Direct;        needsOrderF = false }
        "min",       { pythonFunc = "np.min";            argTransform = Direct;        needsOrderF = false }
        "abs",       { pythonFunc = "np.abs";            argTransform = Direct;        needsOrderF = false }
        "sqrt",      { pythonFunc = "np.sqrt";           argTransform = Direct;        needsOrderF = false }
        "exp",       { pythonFunc = "np.exp";            argTransform = Direct;        needsOrderF = false }
        "log",       { pythonFunc = "np.log";            argTransform = Direct;        needsOrderF = false }
        "disp",      { pythonFunc = "print";             argTransform = Direct;        needsOrderF = false }
        "fprintf",   { pythonFunc = "print";             argTransform = Direct;        needsOrderF = false }
        "sort",      { pythonFunc = "np.sort";           argTransform = Direct;        needsOrderF = false }
        "find",      { pythonFunc = "np.nonzero";        argTransform = Direct;        needsOrderF = false }
        "transpose", { pythonFunc = ".T";                argTransform = AttrStyle "T"; needsOrderF = false }
        "true",      { pythonFunc = "True";              argTransform = Direct;        needsOrderF = false }
        "false",     { pythonFunc = "False";             argTransform = Direct;        needsOrderF = false }
    ]

let tryMapBuiltin (name: string) : BuiltinMapping option =
    match builtinTable.TryGetValue(name) with
    | true, mapping -> Some mapping
    | false, _ -> None
