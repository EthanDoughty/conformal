# Ethan Doughty
# builtins.py
"""Builtin function catalog for MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "acos", "acosh", "accumarray", "all", "any", "asin", "asinh", "atan", "atan2", "atanh",
    "blkdiag",
    "cat", "ceil", "cell", "cell2mat", "cellfun", "char", "chol", "circshift", "complex",
    "cond", "conj", "conv", "cos", "cosh", "cross", "cumprod", "cumsum",
    "deconv", "det", "diag", "diff", "disp", "double",
    "eig", "error", "exp", "expm", "eye",
    "false", "fclose", "fft", "fft2", "fieldnames", "find", "flipud", "fliplr",
    "floor", "fopen", "fprintf", "fread", "fscanf", "full", "fwrite", "fclose",
    "gamrnd",
    "histogram", "horzcat", "hypot",
    "ifft", "ifft2", "imag", "inf", "int16", "int2str", "int32", "int64", "int8",
    "interp1", "interp2", "inv", "iscell", "ischar", "isempty", "isfield",
    "isfinite", "isfloat", "isinf", "isinteger", "islogical", "isnan", "isnumeric",
    "isreal", "isscalar", "issorted", "issparse", "isstring", "isstruct", "issymmetric", "isvector",
    "kron",
    "length", "linspace", "log", "log10", "log2", "logical", "logm", "lu",
    "mat2str", "max", "mean", "median", "min", "mod", "mvnrnd",
    "nan", "ndims", "nnz", "norm", "not", "null", "num2cell", "num2str", "numel",
    "ones", "orth",
    "pinv", "plot", "plot3", "polyfit", "polyval", "power", "prod",
    "qr",
    "rand", "randi", "randn", "rank", "rcond", "real", "rem", "repmat", "reshape", "round",
    "setdiff", "sign", "sin", "single", "sinh", "size", "sort", "sparse", "sprank",
    "sprintf", "sqrt", "sqrtm", "std", "string", "struct", "sub2ind", "sum", "svd",
    "tan", "tanh", "trace", "transpose", "tril", "triu", "true",
    "uint16", "uint32", "uint64", "uint8", "union", "unique",
    "var", "vertcat",
    "warning",
    "xor",
    "zeros",
    # Graphics/plotting — recognized but no shape handler (I/O side effects only)
    "axis", "bar", "box", "cla", "clabel", "clf", "close", "colorbar", "colormap",
    "contour", "contourf", "drawnow", "errorbar", "figure", "fill", "gca", "gcf",
    "grid", "hold", "image", "imagesc", "legend", "light", "line", "loglog",
    "mesh", "meshgrid", "patch", "pause", "pcolor", "quiver", "scatter",
    "semilogx", "semilogy", "set", "get", "shading", "stem", "subplot", "surf",
    "surface", "text", "title", "view", "xlabel", "xlim", "ylabel", "ylim",
    "zlabel", "zlim",
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
    # Domain builtins with shape handlers
    "fft", "ifft", "fft2", "ifft2",  # FFT (same shape)
    "sparse", "full",  # sparsity (passthrough or constructor)
    "cross",  # cross product (passthrough first arg)
    "conv", "deconv",  # convolution (column vector)
    "polyfit",  # polynomial fit (row vector)
    "polyval", "interp1",  # evaluation (same shape as x)
    "meshgrid",  # grid generation (matrix)
    "struct",  # struct constructor (field tracking)
    "fieldnames",  # field names (cell array)
    "ndims",  # dimensionality (scalar)
    "sub2ind",  # subscript to linear index
    "horzcat", "vertcat",  # concatenation builtins
    "pinv",  # pseudoinverse (like inv)
    "expm", "logm", "sqrtm",  # matrix functions (passthrough)
    "circshift",  # circular shift (passthrough)
    "null", "orth",  # null/orthogonal basis (passthrough)
    "isfield",  # struct predicate (scalar)
}
