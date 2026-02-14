# Ethan Doughty
# builtins.py
"""Builtin function catalog for Mini-MATLAB analyzer.

This module defines the set of recognized builtin functions and specifies
which builtins have explicit shape rules in the analyzer.
"""

# Builtins recognized as function calls (not array indexing).
# Sorted alphabetically for maintainability.
KNOWN_BUILTINS = {
    "abs", "det", "diag", "eye", "inv", "isscalar",
    "length", "linspace", "norm", "numel", "ones",
    "rand", "randn", "repmat", "reshape", "size",
    "sqrt", "transpose", "zeros",
}

# Builtins with explicit shape rules (handled in eval_expr_ir).
# Everything else in KNOWN_BUILTINS returns unknown silently.
BUILTINS_WITH_SHAPE_RULES = {
    "zeros", "ones",      # matrix constructors (2-arg form)
    "eye", "rand", "randn",  # matrix constructors (0/1/2-arg forms)
    "abs", "sqrt",        # element-wise (pass through shape)
    "transpose",          # transpose (swap rows/cols)
    "length", "numel",    # query functions (return scalar)
    "size", "isscalar",   # other builtins with shape rules
    "reshape", "repmat",  # matrix manipulation (new)
}
