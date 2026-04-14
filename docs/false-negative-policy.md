# Conformal v3.9.0 False-Negative Policy

This document defines what Conformal guarantees when it reports zero warnings on a MATLAB source file, and what classes of errors it does not detect. It is intended for verification workflows where the scope of analysis must be precisely understood.

## What "zero warnings" means

When Conformal analyzes a file and produces no diagnostics, the following classes of errors are guaranteed absent within the analyzed scope:

1. **Matrix dimension mismatches** in `*`, `+`, `-`, `.*`, `./`, `.^`, `\` between operands whose shapes are both fully resolved (concrete or symbolic).
2. **Concatenation mismatches** where row counts differ in `[A B]` or column counts differ in `[A; B]`, for operands with known shapes.
3. **Reshape element-count violations** where the product of input dimensions provably differs from the product of output dimensions.
4. **Index out-of-bounds** where a subscript's interval is provably outside `[1, dim]` for a matrix with known dimensions.
5. **Type errors** where a non-numeric value (struct, cell, function handle) appears in an arithmetic, transpose, negation, or concatenation context.
6. **Argument count errors** where a function is called with more arguments than its declaration accepts (excluding `varargin` functions and classdef methods with implicit self).

## What "zero warnings" does NOT mean

Conformal is a single-file, single-pass static analyzer. The following classes of errors are outside its scope and will not produce warnings even if present:

### 1. Errors masked by unknown shapes

When a function call is unresolved (the function is not a recognized builtin and not found in the workspace scan), its return value receives `UnknownShape`. Subsequent operations on that value will not produce dimension warnings because `UnknownShape` is the lattice top and absorbs all shape checks conservatively.

**Implication**: A file that calls many unresolved functions may report zero warnings not because the shapes are correct, but because the analyzer lacks sufficient information to check them. The **shape coverage** metric (visible in CLI output when hollowness is detected) quantifies this gap: it reports how many variables have tracked shapes vs. unknown shapes.

### 2. Errors in unanalyzed code paths

Conformal analyzes all branches of `if`/`else`, `switch`/`case`, and `try`/`catch` constructs. However, dead code after an early `return` or `error()` is not analyzed. Code inside `eval()` strings is never analyzed.

### 3. N-dimensional array errors

Conformal tracks matrices as 2-D (`rows x columns`). Operations on 3-D or higher-dimensional arrays (e.g., `A(:,:,k)` on a 3-D array) are handled conservatively: the result shape is unknown, and no dimension checking is performed on the higher dimensions.

### 4. Complex number shape interactions

Complex numbers are treated as scalars or matrices of the same shape as their real counterparts. Conformal does not track whether a value is real or complex, and will not warn about operations that are invalid for complex inputs but valid for real ones.

### 5. Runtime-dependent shapes

Shapes that depend on runtime values not known statically (file I/O results, user input, network data, MEX function returns) are unknown. Operations on these values will not trigger dimension warnings.

### 6. Dynamic dispatch

`eval`, `feval` with non-literal arguments, and function handles stored in variables or data structures are not resolved. Operator overloading in classdef objects is partially supported (method dispatch works for known classes), but operator precedence in complex inheritance hierarchies may not match MATLAB's runtime behavior.

### 7. Simulink and code generation artifacts

Conformal analyzes MATLAB `.m` source files only. Simulink models (`.slx`), Stateflow charts, and generated code are outside scope. MATLAB Coder entry-point constraints (fixed-size declarations via `coder.varsize` or function argument specifications) are not read by Conformal.

### 8. Numeric precision and overflow

Conformal tracks dimensions and shapes, not numeric values. It will not detect overflow, underflow, loss of precision, or ill-conditioned matrix operations.

## Conservative fallback behavior

When Conformal reaches the boundary of its analysis capability, it falls back conservatively:

- Unresolved function calls: return `UnknownShape`, emit `W_UNKNOWN_FUNCTION`
- Unparseable syntax: emit `W_UNSUPPORTED_STMT`, continue analyzing remaining code
- Dynamic field access `s.(expr)`: evaluate to `UnknownShape`
- `eval` / `evalin` / `assignin`: not analyzed, no warning (these are in the suppressed builtins list)
- 3-D array operations: result is `UnknownShape`

In every case, the fallback is to assume unknown rather than to assume correct. This means Conformal will never report a false negative within its analysis scope: if it has enough information to check a shape, it checks it. If it doesn't have enough information, it reports the gap (via `W_UNKNOWN_FUNCTION` or the shape coverage metric) rather than silently passing.

## Determinism

Conformal's analysis is deterministic. The same source file analyzed with the same version and flags will always produce identical output. This property is required for reproducible verification evidence.

## Scope of MATLAB Coder mode (`--coder`)

When `--coder` is enabled, six additional checks fire for constructs that MATLAB Coder cannot compile. These are documented in the Warning Code Catalog. The `--coder` checks are additive: they run after the standard shape analysis and do not affect its behavior.

A clean `--coder --strict` run means: no dimension errors detected AND no Coder-incompatible constructs detected, within the scope described above.

## Relationship to test evidence

Conformal v3.9.0 has 552 integration tests and 28 property-based lattice validation tests (FsCheck). The integration tests use `% EXPECT_WARNING` and `% EXPECT_NO_WARNING` annotations with machine-readable pass/fail semantics. The property-based tests verify join commutativity, associativity, monotonicity, and ordering consistency across randomly generated shapes and intervals.

## Version applicability

This document applies to Conformal v3.9.0 and later. The analysis scope may expand in future versions; this document will be updated accordingly.

## DO-178C qualification

For formal verification workflows that require a DO-330 Tool Operational Requirements document, see [TOR.md](TOR.md). The TOR provides numbered requirements, an anomaly record, and the TQL-5 qualification basis.
