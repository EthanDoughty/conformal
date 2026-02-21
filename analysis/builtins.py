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
    "full", "fullfile", "fgets", "fseek", "ftell", "fwrite", "fclose",
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
    "tan", "tanh", "textscan", "trace", "transpose", "tril", "triu", "true", "typecast",
    "uint16", "uint32", "uint64", "uint8", "union", "unique", "unwrap", "upper",
    "var", "vertcat",
    "warning", "whos", "wishrnd",
    "xor",
    "zeros",
    # Graphics/plotting — recognized but no shape handler (I/O side effects only)
    "autocorr",
    "axis", "bar", "box", "cla", "clabel", "clf", "close", "colorbar", "colormap",
    "contour", "contourf", "drawnow", "errorbar", "ezcontour", "figure", "fill", "gca", "gcf",
    "grid", "hold", "image", "imagesc", "legend", "light", "line", "loglog",
    "maxk", "mesh", "meshgrid", "mnrnd", "mvnpdf", "patch", "pause", "pcolor", "quiver", "scatter",
    "semilogx", "semilogy", "set", "get", "shading", "stem", "subplot", "surf",
    "surface", "text", "title", "view", "xlabel", "xlim", "ylabel", "ylim",
    "zlabel", "zlim",
}
