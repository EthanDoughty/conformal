# Ethan Doughty
# builtins.py
"""Builtin function catalog for Mini-MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "cell", "det", "diag", "eye", "inv", "iscell", "isscalar",
    "length", "linspace", "norm", "numel", "ones",
    "rand", "randn", "repmat", "reshape", "size",
    "sqrt", "transpose", "zeros",
}

# Builtins with explicit shape rules (handled in eval_expr_ir).
# Everything else in KNOWN_BUILTINS returns unknown silently.
BUILTINS_WITH_SHAPE_RULES = {
    "zeros", "ones",      # matrix constructors (1/2-arg forms)
    "eye", "rand", "randn",  # matrix constructors (0/1/2-arg forms)
    "cell",               # cell array constructor (1/2-arg forms)
    "abs", "sqrt",        # element-wise (pass through shape)
    "transpose",          # transpose (swap rows/cols)
    "length", "numel",    # query functions (return scalar)
    "size", "iscell", "isscalar",   # other builtins with shape rules
    "reshape", "repmat",  # matrix manipulation
    "det", "norm",        # scalar-returning operations
    "diag",               # shape-dependent (vector↔diagonal matrix)
    "inv",                # matrix inverse (pass-through for square)
    "linspace",           # row vector generator
}
