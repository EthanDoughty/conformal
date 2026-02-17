# Ethan Doughty
# builtins.py
"""Builtin function catalog for MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "acos", "all", "any", "asin", "atan", "atan2",
    "blkdiag",
    "ceil", "cell", "cos", "cumprod", "cumsum",
    "det", "diag", "diff",
    "exp", "eye",
    "false", "floor",
    "imag", "inf", "inv", "iscell", "ischar", "isempty", "isfinite", "isinf",
    "islogical", "isnan", "isnumeric", "isscalar", "issymmetric",
    "kron",
    "length", "linspace", "log", "log10", "log2",
    "max", "mean", "min", "mod",
    "nan", "norm", "numel",
    "ones",
    "prod",
    "rand", "randn", "real", "rem", "repmat", "reshape", "round",
    "sign", "sin", "size", "sqrt", "sum",
    "tan", "transpose", "true",
    "zeros",
}

# Builtins with explicit shape rules (handled in eval_expr_ir).
# Everything else in KNOWN_BUILTINS returns unknown silently.
BUILTINS_WITH_SHAPE_RULES = {
    "zeros", "ones",      # matrix constructors (1/2-arg forms)
    "eye", "rand", "randn",  # matrix constructors (0/1/2-arg forms)
    "true", "false", "nan", "inf",  # logical/special constructors (0/1/2-arg forms)
    "cell",               # cell array constructor (1/2-arg forms)
    "abs", "sqrt",        # element-wise (pass through shape)
    "sin", "cos", "tan", "asin", "acos", "atan",  # trig (pass through)
    "exp", "log", "log2", "log10",  # exponential/log (pass through)
    "ceil", "floor", "round", "sign",  # rounding (pass through)
    "real", "imag",       # complex (pass through)
    "cumsum", "cumprod",  # cumulative (pass through)
    "transpose",          # transpose (swap rows/cols)
    "length", "numel",    # query functions (return scalar)
    "size", "iscell", "isscalar",   # other builtins with shape rules
    "isempty", "isnumeric", "islogical", "ischar",  # type predicates (return scalar)
    "isnan", "isinf", "isfinite", "issymmetric",  # value predicates (return scalar)
    "reshape", "repmat",  # matrix manipulation
    "det", "norm",        # scalar-returning operations
    "diag",               # shape-dependent (vector↔diagonal matrix)
    "inv",                # matrix inverse (pass-through for square)
    "linspace",           # row vector generator
    "sum", "prod", "mean", "any", "all",  # reductions
    "min", "max",         # min/max (reduction or elementwise)
    "mod", "rem", "atan2",  # elementwise 2-arg
    "diff",               # differentiation (dimension subtraction)
    "kron",               # Kronecker product
    "blkdiag",            # block diagonal concatenation
}
