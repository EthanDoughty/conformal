# Ethan Doughty
# builtins.py
"""Builtin function catalog for MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "acos", "acosh", "accumarray", "addpath", "all", "angle", "any", "arrayfun",
    "asin", "asinh", "assert", "atan", "atan2", "atanh",
    "bicgstab", "bitand", "bitshift", "bitor", "bitxor", "blkdiag",
    "cat", "cd", "ceil", "cell", "cell2mat", "cell2struct", "cellfun", "cgs", "char", "chol",
    "circshift", "class", "complex", "cond", "conj", "contains", "conv", "cos", "cosh",
    "cross", "cumprod", "cumsum",
    "datestr", "dbstack", "deal", "deblank", "dec2hex", "deconv", "deg2rad", "delete",
    "det", "diag", "diff", "dir", "disp", "display", "double",
    "eig", "eigs", "error", "eval", "exist", "exp", "expm", "eye",
    "false", "fclose", "feval", "fft", "fft2", "fftshift", "fieldnames", "fileparts",
    "find", "flipud", "fliplr", "floor", "fopen", "fprintf", "fread", "fscanf",
    "full", "fullfile", "fwrite", "fclose",
    "gamrnd", "getfield", "gmres",
    "hex2dec", "hex2num", "histogram", "horzcat", "hypot",
    "ifft", "ifft2", "ifftshift", "imag", "inf", "Inf", "input", "int16", "int2str",
    "int32", "int64", "int8", "interp1", "interp2", "intersect", "inv", "iscell",
    "ischar", "isempty", "isfield", "isfinite", "isfloat", "isinf", "isinteger",
    "islogical", "ismember", "isnan", "isnumeric", "isreal", "isscalar", "issorted",
    "issparse", "isstring", "isstruct", "issymmetric", "isvector",
    "kron",
    "length", "linspace", "load", "log", "log10", "log2", "logical", "logm", "logspace",
    "lower", "lsqnonneg", "lu",
    "mat2cell", "mat2str", "max", "mean", "median", "min", "mink", "mkdir", "mod",
    "mvnrnd",
    "nan", "NaN", "nanmax", "nanmean", "nanmin", "nanstd", "nansum", "nargin",
    "nargout", "ndgrid", "ndims", "nnz", "norm", "normpdf", "not", "null", "num2cell",
    "num2hex", "num2str", "numel",
    "ones", "orderfields", "orth",
    "pcg", "permute", "pinv", "plot", "plot3", "poly", "polyfit", "polyval", "power",
    "ppval", "print", "prod",
    "qr",
    "rad2deg", "rand", "randi", "randn", "rank", "rcond", "real", "regexp", "regexpi",
    "rem", "repmat", "reshape", "rmpath", "roots", "round",
    "save", "saveas", "setdiff", "setfield", "setxor", "sgolayfilt", "shiftdim", "squeeze",
    "sign", "sin", "single", "sinh", "size", "sort", "sparse", "spline", "sprank",
    "sprintf", "sqrt", "sqrtm", "std", "str2double", "strcmp", "strcmpi", "strfind",
    "strjoin", "strmatch", "strrep", "strsplit", "strtrim", "string", "struct",
    "struct2cell", "structfun", "sub2ind", "sum", "svd", "svds",
    "tan", "tanh", "trace", "transpose", "tril", "triu", "true", "typecast",
    "uint16", "uint32", "uint64", "uint8", "union", "unique", "unwrap", "upper",
    "var", "vertcat",
    "warning", "whos", "wishrnd",
    "xor",
    "zeros",
    # Graphics/plotting — recognized but no shape handler (I/O side effects only)
    "axis", "bar", "box", "cla", "clabel", "clf", "close", "colorbar", "colormap",
    "contour", "contourf", "drawnow", "errorbar", "figure", "fill", "gca", "gcf",
    "grid", "hold", "image", "imagesc", "legend", "light", "line", "loglog",
    "maxk", "mesh", "meshgrid", "patch", "pause", "pcolor", "quiver", "scatter",
    "semilogx", "semilogy", "set", "get", "shading", "stem", "subplot", "surf",
    "surface", "text", "title", "view", "xlabel", "xlim", "ylabel", "ylim",
    "zlabel", "zlim",
}

# Builtins with explicit shape rules (handled in eval_expr_ir).
# Everything else in KNOWN_BUILTINS returns unknown silently.
BUILTINS_WITH_SHAPE_RULES = {
    "zeros", "ones",      # matrix constructors (1/2-arg forms)
    "eye", "rand", "randn", "randi",  # matrix constructors (0/1/2-arg forms)
    "true", "false", "nan", "inf", "NaN", "Inf",  # logical/special constructors (0/1/2-arg forms)
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
    "linspace", "logspace",  # row vector generators
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
    "fftshift", "ifftshift",  # FFT shift (passthrough)
    "sgolayfilt", "squeeze", "unwrap", "deg2rad", "rad2deg", "angle",  # passthrough
    "typecast",  # type cast (passthrough)
    "nanmean", "nansum", "nanstd", "nanmin", "nanmax",  # NaN-ignoring reductions
    "fullfile",  # string-returning path join
    "strcmpi", "strcmp", "exist", "str2double",  # scalar predicates/queries
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
