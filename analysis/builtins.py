# Ethan Doughty
# builtins.py
"""Builtin function catalog for MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "acos", "acosh", "all", "any", "asin", "asinh", "atan", "atan2", "atanh",
    "blkdiag",
    "cat", "ceil", "cell", "char", "chol", "complex", "cond", "conj", "cos", "cosh", "cumprod", "cumsum",
    "det", "diag", "diff", "disp", "double",
    "eig", "error", "exp", "eye",
    "false", "find", "flipud", "fliplr", "floor", "fprintf",
    "hypot",
    "imag", "inf", "int16", "int2str", "int32", "int64", "int8", "inv", "iscell", "ischar",
    "isempty", "isfinite", "isfloat", "isinf", "isinteger", "islogical", "isnan", "isnumeric",
    "isreal", "isscalar", "issorted", "issparse", "isstring", "isstruct", "issymmetric", "isvector",
    "kron",
    "length", "linspace", "log", "log10", "log2", "logical", "lu",
    "mat2str", "max", "mean", "median", "min", "mod",
    "nan", "nnz", "norm", "not", "num2str", "numel",
    "ones",
    "power", "prod",
    "qr",
    "rand", "randi", "randn", "rank", "rcond", "real", "rem", "repmat", "reshape", "round",
    "sign", "sin", "single", "sinh", "size", "sort", "sprank", "sprintf", "sqrt", "std", "string", "sum", "svd",
    "tan", "tanh", "trace", "transpose", "tril", "triu", "true",
    "uint16", "uint32", "uint64", "uint8", "unique",
    "var",
    "warning",
    "xor",
    "zeros",
}

# Builtins with explicit shape rules (handled in eval_expr_ir).
# Everything else in KNOWN_BUILTINS returns unknown silently.
BUILTINS_WITH_SHAPE_RULES = {
    "zeros", "ones",      # matrix constructors (1/2-arg forms)
    "eye", "rand", "randn", "randi",  # matrix constructors (0/1/2-arg forms)
    "true", "false", "nan", "inf",  # logical/special constructors (0/1/2-arg forms)
    "cell",               # cell array constructor (1/2-arg forms)
    "abs", "sqrt",        # element-wise (pass through shape)
    "sin", "cos", "tan", "asin", "acos", "atan",  # trig (pass through)
    "tanh", "cosh", "sinh", "atanh", "acosh", "asinh",  # hyperbolic trig (pass through)
    "conj", "not",        # complex / logical (pass through)
    "flipud", "fliplr", "triu", "tril",  # flip/triangular (pass through)
    "sort", "unique",     # sort/unique (pass through)
    "exp", "log", "log2", "log10",  # exponential/log (pass through)
    "ceil", "floor", "round", "sign",  # rounding (pass through)
    "real", "imag",       # complex (pass through)
    "cumsum", "cumprod",  # cumulative (pass through)
    "transpose",          # transpose (swap rows/cols)
    "length", "numel",    # query functions (return scalar)
    "size", "iscell", "isscalar",   # other builtins with shape rules
    "isempty", "isnumeric", "islogical", "ischar",  # type predicates (return scalar)
    "isnan", "isinf", "isfinite", "issymmetric",  # value predicates (return scalar)
    "isstruct", "isreal", "issparse", "isvector", "isinteger", "isfloat", "isstring", "issorted",  # more predicates
    "reshape", "repmat",  # matrix manipulation
    "det", "norm",        # scalar-returning operations
    "trace", "rank", "cond", "rcond", "nnz", "sprank",  # more scalar queries
    "diag",               # shape-dependent (vector↔diagonal matrix)
    "inv",                # matrix inverse (pass-through for square)
    "linspace",           # row vector generator
    "sum", "prod", "mean", "any", "all",  # reductions
    "median", "var", "std",  # more reductions
    "min", "max",         # min/max (reduction or elementwise)
    "mod", "rem", "atan2",  # elementwise 2-arg
    "power", "hypot", "xor",  # more elementwise 2-arg
    "diff",               # differentiation (dimension subtraction)
    "kron",               # Kronecker product
    "blkdiag",            # block diagonal concatenation
    "cat",                # concatenation along dimension
    "find",               # find indices (row vector, unknown length)
    "double", "single", "int8", "int16", "int32", "int64",  # type casts
    "uint8", "uint16", "uint32", "uint64", "logical", "complex",  # more type casts
    "num2str", "int2str", "mat2str", "char", "string", "sprintf",  # string returns
    "eig", "svd", "lu", "qr", "chol",  # linear algebra (single + multi-return)
}
