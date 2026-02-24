<div align="center">

# Conformal

### Static Shape & Dimension Analysis for MATLAB

[![Version](https://img.shields.io/badge/version-2.0.0-orange.svg)](#motivation-and-future-directions)
[![VS Code](https://img.shields.io/badge/VS%20Code-Marketplace-007ACC.svg)](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4.svg)](https://dotnet.microsoft.com/download)
[![Tests](https://img.shields.io/badge/tests-354%20passing-brightgreen.svg)](#test-suite)
[![License](https://img.shields.io/badge/license-BSL--1.1-purple.svg)](LICENSE)

*Matrices must be **conformable** before they can perform. Conformal makes sure they are.*

</div>

---

Conformal catches matrix dimension errors in MATLAB code before you run it. You write `A * B` where the inner dimensions don't match, and instead of finding out at runtime, Conformal tells you at analysis time. It tracks shapes through assignments, function calls, control flow, loops, and symbolic dimensions, all without needing MATLAB installed.

```matlab
A = zeros(3, 4);
B = ones(5, 2);
C = A * B;
D = [A; B];
```

```
Warnings:
  - Line 3: Dimension mismatch in expression (A * B):
    inner dims 4 vs 5 (shapes matrix[3 x 4] and matrix[5 x 2])
  - Line 4: Vertical concatenation requires equal column counts
    across rows; got 4 and 2 in matrix literal.

Final environment:
    Env{A: matrix[3 x 4], B: matrix[5 x 2], C: unknown, D: unknown}
```

## Quick Start

**VS Code** (The recommended option): Install Conformal from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal) by searching "Conformal" in Extensions, or run the following command:
```bash
code --install-extension EthanDoughty.conformal
```
Open any `.m` file and the diagnostics appear as underlines. Hover a variable to see its inferred shape. No configuration is needed.

**CLI**: Requires [.NET 8.0 SDK](https://dotnet.microsoft.com/download) or later. Again, no MATLAB installation is required.
```bash
git clone https://github.com/EthanDoughty/conformal.git
cd conformal/src
dotnet run -- ../tests/basics/inner_dim_mismatch.m
```

## Performance

The single-file analysis takes under 100ms, even for 700-line files with 36 warnings, and the cross-file workspace analysis runs in about 70ms. The full test suite (354 tests total) finishes in about one second, with no MATLAB runtime involved during any part of the process.

The VS Code extension runs the analyzer while you are typing code, since it is compiled to JavaScript, using the Fable tool, so there is no subprocess startup cost and analysis works on every keystroke with a 500ms debounce.

## What the Analysis Detects

All warnings include source line numbers. When Conformal finds a definite error, it marks the result as unknown and keeps going so you get as many diagnostics as possible in a single pass.

By default, Conformal shows only high-confidence warnings. These include dimension mismatches, type errors, bounds violations, division by zero, and constraint conflicts. There are 19 warning codes that only appear in the --strict mode, mostly things that are low-confidence, like `W_UNKNOWN_FUNCTION`, `W_STRUCT_FIELD_NOT_FOUND`, and `W_SUSPICIOUS_COMPARISON`. The idea is that you can run default mode in CI without false-positive noise, and use --strict when you want a full picture analysis.

### Operations

Conformal detects dimension mismatches in `+`, `-`, `*`, `.*`, `./`, `^`, `.^`. Scalar-matrix broadcasting (e.g. `s*A`, `s + A`) is handled. Backslash `A\b` (mldivide) follows the same inner-dimension logic that multiplication does. Element-wise logical `&` and `|` pass shapes through like an element-wise op usually would. Logical NOT `~` and dot-transpose `.'` also carry shapes through. When you use `*` where `.*` was probably intended, Conformal suggests the fix.

### Literals and concatenation

Conformal handles matrix literals like `[1 2; 3 4]`, cell array literals like `{1, 2; 3, 4}`, and string literals in both quote styles (`'hello'`, `"world"`). Horizontal concatenation `[A B]` checks that row counts match, and vertical concatenation `[A; B]` checks columns. Symbolic dimensions compose through concatenation, so `[A B]` where A is `n x k` and B is `n x m` gives `n x (k+m)`. An empty matrix `[]` is treated as the identity element for concatenation, so `[[] x]` and `[[] ; x]` both simplify to `x` with no false mismatch warning. Matrix literal spacing is handled correctly too, so `[1 -2; 3 -4]` parses as four elements rather than as subtraction.

### Indexing

Conformal supports parenthesized indexing `A(i,j)`, slice indexing `A(:,j)` and `A(i,:)`, range indexing `A(2:5,:)`, linear indexing, and full-matrix `A(:,:)`. Curly-brace indexing `C{i,j}` works for cell arrays with per-element shape tracking, where a literal index extracts the precise element shape. The `end` keyword works in indexing contexts with arithmetic support (`C{end}`, `C{end-1}`, `A(1:end, 2)`, `A(end-2:end, :)`). Indexed assignment `M(i,j) = expr` preserves the matrix shape, and the write side doesn't check bounds, since MATLAB auto-expands arrays on assignment. Read-side out-of-bounds checking is unchanged.

### Functions

Over 635 MATLAB builtins are recognized (so calls to them don't produce `W_UNKNOWN_FUNCTION` warnings), and around 315 of those have explicit shape rules. First, matrix constructors like `zeros`, `ones`, `eye`, `rand`, `randn`, `true`, `false`, `nan`, and `inf` handle all three call forms: no args gives a scalar, one arg gives an n x n square, and two args give an m x n matrix. Second, shape transformations include `reshape` (with a conformability check), `repmat`, `diag`, `transpose`, `horzcat`, `vertcat`, and more specialized ones like `kron` (where `kron(A[m x n], B[p x q])` gives `matrix[(m*p) x (n*q)]`) and `blkdiag` (where `blkdiag(A[m x n], B[p x q])` gives `matrix[(m+p) x (n+q)]`). Third, element-wise math functions like `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `ceil`, `floor`, and their relatives all pass the input shape through. Reductions like `sum`, `prod`, `mean`, `min`, `max`, and `diff` accept an optional dimension argument. Finally, type predicates like `isscalar`, `iscell`, `isempty`, and `isnumeric` return scalar, and linear algebra functions like `det`, `inv`, `norm`, and `linspace`, query functions like `size`, `length`, and `numel`, and two-argument element-wise functions like `mod`, `rem`, and `atan2` are also covered.

Dimension arithmetic works inside builtin arguments, so `zeros(n+1, 2*m)` is tracked symbolically.

Conformal analyzes user-defined functions at each call site with the caller's argument shapes. Three forms are supported: single return (`function y = f(x)`), multi-return (`function [a, b] = f(x)`), and procedures (including no-arg `function name` syntax). Nested `function...end` blocks inside another function body are also supported, with read/write access to the parent workspace via scope chains and forward-reference visibility between siblings. The parser also handles pre-2016 end-less function definitions (files where `end` is omitted), and space-separated multi-return syntax (`function [a b c] = f(...)` without commas). Anonymous functions `@(x) expr` are analyzed the same way, with by-value closure capture at definition time. Function handles `@funcName` dispatch to their targets. Results are cached per argument shape tuple so the same function called with the same shapes isn't re-analyzed.

When analyzing a file, Conformal also scans sibling `.m` files in the same directory and fully analyzes their bodies (parse -> analyze) to infer real return shapes. Dimension aliasing works across file boundaries, subfunctions in external files are supported, and cross-file cycles (A->B->A) are detected and handled gracefully. Unparseable external files emit `W_EXTERNAL_PARSE_ERROR`.

### Data structures

Conformal tracks struct field assignment (`s.x = A`), field access, and chained dot notation (`s.x.y`). Missing field access on a known struct emits a warning. Struct shapes join across branches by taking the union of fields. When you assign a field to a variable whose base shape is unknown, for example the return value of an unrecognized function, Conformal creates an open struct, written `struct{x: matrix[1 x 3], ...}`. Open structs don't warn on missing field access, since there could be more fields that aren't tracked. Multi-return destructuring supports dotted targets, so `[s.x, s.y] = get_pair()` works and populates the struct's field map. The lattice ordering is bottom < closed struct < open struct < unknown.

Cell arrays work with `cell(n)` and `cell(m,n)` constructors, curly-brace indexing, and element assignment. Literal indexing `C{i}` extracts the precise shape of element `i` when available, and dynamic indexing joins all element shapes conservatively. Curly-brace indexing on a non-cell emits a warning.

### Control flow

Conformal covers `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`/`otherwise`, `try`/`catch`, `break`, `continue`, and `return`. When branches assign different shapes to the same variable, Conformal joins them conservatively. Loops use a single pass by default, or widening-based fixed-point iteration via `--fixpoint` for guaranteed convergence in at most 2 iterations. In `--fixpoint` mode, for-loop accumulation patterns like `A = [A; delta]` and `A = [A, delta]` are detected and refined algebraically: the iteration count is extracted from the range (`(b-a)+1`), and the widened dimension is replaced with `init_dim + iter_count * delta_dim`. Interval widening in fixpoint loops produces sound post-loop intervals, so interval-based checks remain accurate even after a fixpoint run.

Switch/case bodies also benefit from interval refinement: when the switch expression is an integer variable and a case arm matches a concrete value, Conformal narrows the variable to that value inside the case body, which can eliminate false-positive dimension and bounds warnings in dispatch-style code.

### Symbolic dimensions

Variables with unknown concrete size get symbolic names like `n`, `m`, `k`, and those names propagate through operations: `1:n` gives a `1 x n` vector, `[A B]` computes `n x (k+m)`, `zeros(n+1, 2*m)` tracks the arithmetic. Symbolic dimensions are represented as canonical polynomials with rational coefficients, so `n+m` and `m+n` are recognized as equal, and `n+n` simplifies to `2*n`. When a function is called with a symbolic argument `n`, the dimension name propagates into the function body.

### Interval analysis

In parallel with shape inference, Conformal tracks scalar integer variables through an integer interval domain `[lo, hi]`. This is what enables three additional checks: `W_DIVISION_BY_ZERO` when the divisor is provably zero, `W_INDEX_OUT_OF_BOUNDS` when an index is provably outside the matrix dimensions, and `W_POSSIBLY_NEGATIVE_DIM` when a dimension expression is provably non-positive.

Initially, for-loop variables are bound to their range interval, so `for i = 1:n` records `i` in `[1, n]` with a symbolic upper bound. Comparisons against symbolic bounds fall back soundly, and intervals join conservatively across control-flow branches.

Additionally, branch conditions narrow variable intervals inside the branch body. If you write `if x > 0`, Conformal refines `x` to `[1, +inf]` for the true branch, which can eliminate false-positive out-of-bounds and negative-dim warnings when a guard proves safety. Conformal supports `>`, `>=`, `<`, `<=`, `==`, `~=`, compound `&&` conditions, and operator flipping like `5 >= x`.

### Type errors

When you use a non-numeric type (struct, cell, function_handle) where a numeric value is expected, Conformal emits a type mismatch error. Arithmetic operations like `+`, `-`, `*`, and `.*` on structs or cells emit `W_ARITHMETIC_TYPE_MISMATCH`. Transpose on a non-numeric type emits `W_TRANSPOSE_TYPE_MISMATCH`, negation emits `W_NEGATE_TYPE_MISMATCH`, and mixing incompatible types in a matrix literal (like `[s, A]` where `s` is a struct) emits `W_CONCAT_TYPE_MISMATCH`. All four codes are Error severity, not warnings.

### Witness generation

For dimension conflict warnings, Conformal can optionally produce a concrete counterexample proving the bug is real. A witness is a set of variable assignments like `n=3, m=5` under which the conflicting dimensions evaluate to different integers. The solver pulls from variables with known concrete values, interval bounds from interval analysis, and branch path conditions to narrow the candidate space before enumerating. It bails out conservatively when dimensions are unknown, when symbolic terms are quadratic or higher, or when more than 8 free variables are involved.

In `--witness enrich` mode (the default when `--witness` is given), the concrete assignment is printed below each warning. In `--witness filter` mode, only warnings with a verified witness are shown, which gives you zero false positives. In `--witness tag` mode, each warning is prefixed with `[confirmed]` or `[possible]`. The LSP server always runs witness generation and appends the witness explanation to the hover text automatically, with no extra configuration needed.

## Language Coverage

Conformal parses and tracks shapes through:

| Category | Constructs |
|----------|-----------|
| Expressions | `+`, `-`, `*`, `.*`, `./`, `^`, `.^`, `\`, `&`, `\|`, `==`, `~=`, `<`, `>`, `<=`, `>=`, `&&`, `\|\|`, `~`, `'`, `.'` |
| Literals | `[1 2; 3 4]`, `{1, 2; 3, 4}`, `'string'`, `"string"`, `1:n` |
| Indexing | `A(i,j)`, `A(:,j)`, `A(2:5,:)`, `C{i}`, `C{i} = x` |
| Assignment | `x = expr`, `s.field = expr`, `C{i} = expr`, `M(i,j) = expr`, `[a, b] = f(x)`, `[~, b] = f(x)`, `[s.x, s.y] = f(x)` |
| Functions | `function y = f(x)`, `function name` (no-arg), nested `function` blocks, `@(x) expr`, `@funcName`, 635 recognized builtins (315 with shape rules) |
| Control flow | `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`, `try`/`catch` |
| Statements | `break`, `continue`, `return` |
| Data types | scalars, matrices, strings, structs, cell arrays, function handles |

## Shape System

Every expression Conformal analyzes gets a shape from an abstract domain. There are 7 user-visible shapes (plus an internal `bottom` for unbound variables):

| Shape | Example | Notes |
|-------|---------|-------|
| `scalar` | `5`, `x` | Single numeric value |
| `matrix[r x c]` | `matrix[3 x 4]`, `matrix[n x m]` | Dimensions can be concrete, symbolic, or unknown |
| `string` | `'hello'` | Char array |
| `struct{...}` | `struct{x: scalar, ...}` (open), `struct{x: scalar}` (closed) | Tracks fields and their shapes; open structs (from unknown bases) suppress missing-field warnings |
| `function_handle` | `@(x) x'`, `@sin` | Tracks lambda ID for join precision |
| `cell[r x c]` | `cell[3 x 1]` | Cell array with optional per-element shape tracking |
| `unknown` | | Error or indeterminate; the lattice top |

The shape domain is implemented as a discriminated union in F# (`Shapes.fs`), with `bottom` serving as the lattice identity, used internally for unbound variables and never surfaced in output.

Dimensions in `matrix[r x c]` can be concrete integers like `3` or `100`, symbolic names like `n`, `m`, `k`, symbolic expressions like `n+m`, `2*k`, or `n+1`, or simply unknown when no information is available.

Symbolic dimensions use a polynomial representation with rational coefficients. Canonicalization handles commutativity (`n+m` = `m+n`), like-term collection (`n+n` = `2*n`), and constant-offset conflict detection. When control flow branches assign conflicting dimensions to the same variable, Conformal joins them to unknown. In loops with `--fixpoint`, conflicting dimensions get widened to unknown while stable dimensions are preserved.

## Project Structure

```
src/                    F# analyzer (lexer, parser, shape inference, builtins, diagnostics, LSP server)
vscode-conformal/       VS Code extension (TypeScript client + Fable-compiled analyzer)
  fable/                Fable compilation project (F# to JavaScript, shares src/*.fs files)
  src/                  TypeScript extension and LSP server code
tests/                  354 self-checking MATLAB programs in 17 categories
.github/                CI workflow (build, test, compile Fable, package VSIX)
```

## Test Suite

Conformal is validated by 354 self-checking MATLAB programs organized into 17 categories. Each test embeds its expected behavior as inline assertions:

```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]   % Override for --fixpoint mode
```

The test runner checks that Conformal's output matches these expectations. In addition to the `.m` test files, the shape domain is validated by 28 property-based tests using FsCheck, covering 6 sections of the lattice (join commutativity, associativity, monotonicity, and lattice ordering for shapes and intervals). These run as part of `dotnet run -- --tests`.

---

<details open>
<summary><h3>Basics (17 tests)</h3></summary>

Foundation tests for core matrix operations and dimension compatibility.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `valid_add.m` | Conformable matrix addition succeeds | 0 |
| `invalid_add.m` | Dimension mismatch in addition detected | 1 |
| `matrix_multiply.m` | Valid matrix multiplication (inner dims match) | 0 |
| `inner_dim_mismatch.m` | Catch incompatible inner dimensions in `A * B` | 1 |
| `scalar_matrix_ops.m` | Scalar-matrix broadcasting works correctly | 0 |
| `elementwise_ops.m` | Shape mismatch in element-wise operations flagged | 1 |
| `reassignment.m` | Incompatible variable reassignment detected | 1 |
| `type_errors.m` | Type mismatch warnings for struct/cell/function_handle in arithmetic, transpose, negation, and concat | 4 |
| `power_ops.m` | Matrix power `^` and element-wise power `.^` shape rules | 1 |
| `backslash.m` | Backslash (mldivide) `A\b` shape inference | 1 |
| `dot_transpose.m` | Non-conjugate transpose `.'` returns transposed shape | 0 |
| `elementwise_logical.m` | Element-wise logical `&` and `\|` return scalar or matrix shape | 0 |
| `logical_not.m` | Logical NOT `~x` shape passthrough | 0 |
| `tilde_unused.m` | Tilde `[~, x] = f()` as unused output placeholder | 0 |
| `space_destructure.m` | Space-separated destructuring `[a b] = expr` without commas | 0 |
| `scientific_notation.m` | Scientific notation literals (`1e6`, `1.5e-3`, `1E+9`) parsed as scalars | 0 |
| `matrix_literal_transpose.m` | Transpose of a matrix literal `[1 2; 3 4]'` parses correctly and returns transposed shape | 0 |

</details>

<details open>
<summary><h3>Symbolic Dimensions (6 tests)</h3></summary>

Tests symbolic dimension tracking, arithmetic, and canonical polynomial representation.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `dimension_tracking.m` | Colon vectors preserve symbolic dimensions (`1:n`) | 0 |
| `dimension_arithmetic.m` | Addition/multiplication of symbolic dimensions (e.g., `n+m`, `2*k`) | 0 |
| `canonicalization.m` | SymDim polynomial canonicalization ensures `zeros(n,m)` joins with `zeros(n,m)` | 0 |
| `commutativity_join.m` | Commutative equality: `(n+m)` joins with `(m+n)` | 0 |
| `like_terms.m` | Like-term collection: `(n+n)` canonicalizes to `(2*n)` | 0 |
| `rational_dimensions.m` | Rational coefficients in symbolic dimensions (e.g., `n/2`) | 0 |

>Symbolic dimensions are represented as canonical polynomials with rational coefficients, enabling precise tracking of parametric shapes across function boundaries.

</details>

<details open>
<summary><h3>Indexing (17 tests)</h3></summary>

MATLAB-style indexing including scalar, slice, range, linear indexing, and `end` keyword arithmetic.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `scalar_index.m` | Scalar indexing `A(i,j)` returns scalar | 0 |
| `slice_index.m` | Slice indexing `A(:,j)` and `A(i,:)` shape rules | 0 |
| `range_index.m` | Range indexing `A(2:5, :)` preserves symbolic dimensions | 0 |
| `linear_index.m` | Linear indexing on scalar base is an error | 1 |
| `invalid_row_index.m` | Constant-range row indexing edge cases | 0 |
| `invalid_col_index.m` | Constant-range column indexing edge cases | 0 |
| `invalid_linear_index.m` | Non-scalar index argument flagged | 1 |
| `end_arithmetic_matrix.m` | `end` keyword arithmetic in array indexing (`end-1`, `end-2:end`) | 0 |
| `symbolic_range.m` | Symbolic range indexing: variable endpoint `A(1:k,:)` -> `k x c` extent | 0 |
| `end_position.m` | `end` resolves to column dimension in column position (non-square matrix) | 0 |
| `symbolic_end_range.m` | `end` on symbolic matrices: `A(1:end,:)` -> `n x m`, `A(1:end-1,:)` -> `(n-1) x m` | 0 |
| `index_assign.m` | Basic indexed assignment `M(i,j) = expr` preserves matrix dimensions | 0 |
| `index_assign_bounds.m` | MATLAB auto-expands arrays on write; no `W_INDEX_OUT_OF_BOUNDS` on indexed assignment | 0 |
| `index_assign_loop.m` | Indexed assignment inside for loop body | 0 |
| `index_assign_in_function.m` | Indexed assignment in function body; caller sees correct return shape | 0 |
| `matrix_literal_index.m` | Matrix literal as index argument `A([1 2 3])` returns the correct shape; colon ranges inside `[...]` index args work | 0 |
| `colon_in_matrix_index.m` | Colon range inside a matrix literal index arg `A([1:3])` parses correctly via `colon_visible` parameter threading | 0 |

</details>

<details open>
<summary><h3>Control Flow (17 tests)</h3></summary>

Control-flow join semantics for if/elseif/else, switch/case, try/catch, break, and continue.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `if_branch_mismatch.m` | Conservative join when branches assign conflicting shapes | 1 |
| `if_else_error_branch.m` | Error in one branch propagates as `unknown` | 1 |
| `suspicious_comparison.m` | Matrix-scalar comparisons flagged (likely bug) | 1 |
| `elseif_chain.m` | If-elseif-else chain joins all branch environments | 0 |
| `elseif_no_else.m` | Elseif without else joins with pre-condition environment | 0 |
| `elseif_mismatch.m` | Shape conflicts across elseif branches handled conservatively | 0 |
| `break_simple.m` | Break statement exits for loop correctly | 0 |
| `continue_simple.m` | Continue skips to next while loop iteration | 0 |
| `break_nested_loop.m` | Break only exits innermost loop | 0 |
| `switch_basic.m` | Switch/case with otherwise joins all branches | 0 |
| `switch_no_otherwise.m` | Switch without otherwise joins cases with pre-switch env | 0 |
| `switch_mismatch.m` | Conflicting shapes across cases joined conservatively | 0 |
| `try_catch_basic.m` | Try/catch joins try-branch with catch-branch | 1 |
| `try_catch_no_error.m` | Catch block unused when no error in try block | 0 |
| `try_nested.m` | Nested try/catch blocks work correctly | 0 |
| `break_in_if_else.m` | Break in if-inside-loop correctly propagates; else branch shape used when break taken | 0 |
| `try_catch_break.m` | Break inside try/catch block propagates correctly out of the enclosing loop | 0 |

>Conservative join semantics. When branches disagree on a variable's shape, the analyzer joins to the least upper bound (often `unknown`).

</details>

<details>
<summary><h3>Literals (11 tests)</h3></summary>

Matrix literals, string literals, and concatenation constraints.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `matrix_literal.m` | Basic matrix literals `[1 2; 3 4]` parsed and shaped correctly | 0 |
| `horzcat_vertcat.m` | Horizontal/vertical concatenation dimension constraints | 1 |
| `symbolic_concat.m` | Symbolic dimension addition in concatenation (e.g., `[A B]` -> `n x (k+m)`) | 0 |
| `string_literal.m` | String literals with both quote styles (`'foo'`, `"bar"`) | 0 |
| `string_horzcat.m` | String concatenation via horizontal concatenation | 0 |
| `string_matrix_error.m` | String-matrix arithmetic operations flagged | 1 |
| `string_in_control_flow.m` | String/scalar shape joins across branches | 0 |
| `matrix_spacing.m` | Matrix literal spacing: `[1 -2]` is two elements (not subtraction) | 0 |
| `cell_spacing.m` | Cell literal spacing disambiguation (`{1 -2}` is two elements) | 0 |
| `empty_concat.m` | `[]` is identity for concatenation: `[[] x]` -> `x`, `[[] ; x]` -> `x` | 1 |
| `negative_matrix_elements.m` | Matrix literals with negative elements: `[-1 -2]` is `1x2`, not a scalar | 0 |

</details>

<details>
<summary><h3>Builtins (27 tests)</h3></summary>

Shape rules for 635 recognized MATLAB builtins (315 with shape handlers), call/index disambiguation, and dimension arithmetic.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `unknown_function.m` | Unknown function calls emit warning | 1 |
| `shape_preserving.m` | `size()` and `isscalar()` return scalar | 0 |
| `call_vs_index.m` | Apply node disambiguation (function call vs indexing) | 1 |
| `apply_disambiguation.m` | Environment-based disambiguation prefers bound variables | 0 |
| `constructors.m` | Matrix constructors (`zeros`, `ones`, `eye`, `rand`) and element-wise ops | 0 |
| `dim_arithmetic.m` | Dimension arithmetic in builtin args: `zeros(n+1, 2*m)` | 0 |
| `reshape_repmat.m` | `reshape` and `repmat` shape transformations | 0 |
| `remaining_builtins.m` | `det`, `diag`, `inv`, `linspace`, `norm` shape rules | 0 |
| `type_queries.m` | Type query functions: `iscell()`, `isscalar()` return scalar | 0 |
| `elementwise_math.m` | Element-wise math functions (trig, exp/log, rounding): pass-through shape | 0 |
| `reductions.m` | Reduction functions (`sum`, `prod`, `mean`, `min`, `max`, `diff`) with dimension args | 0 |
| `constructors_logical.m` | Logical/special constructors (`true`, `false`, `nan`, `inf`) | 0 |
| `type_predicates_extended.m` | Type predicates (`isempty`, `isnumeric`, `isnan`, `issymmetric`, etc.) | 0 |
| `reshape_conformability.m` | `reshape` conformability check: element count mismatch emits `W_RESHAPE_MISMATCH` | 1 |
| `kron_blkdiag.m` | `kron` (Kronecker product) and `blkdiag` (variadic block diagonal) shape rules | 0 |
| `expanded_builtins.m` | Coverage across all handler categories (hyperbolic trig, type casts, string returns, etc.) | 0 |
| `multi_return_builtins.m` | Multi-return builtins like `eig`, `svd`, `sort`, `find` with `[a, b] = f(x)` syntax | 0 |
| `math_constants.m` | Predefined constants (`pi`, `eps`, `inf`, `nan`, `i`) recognized as scalars | 0 |
| `struct_constructor.m` | `struct()` constructor with field names and values; field tracking | 0 |
| `domain_builtins.m` | Domain builtins: `fft`/`ifft` (passthrough), `polyfit` (row vector), `polyval` (input shape), `ndims` (scalar) | 0 |
| `range_args.m` | Builtins receiving colon-range arguments like `polyval(p, 1:10)` don't crash | 0 |
| `corpus_builtins.m` | Dogfood corpus builtins recognized without `W_UNKNOWN_FUNCTION` (NaN/Inf variants, string ops, nan-ignoring reductions, I/O ops) | 0 |
| `expanded_builtins_2.m` | File I/O builtins (`fopen`, `fgets`, `fseek`, `ftell`, `textscan`, `fclose`) recognized without warnings | 0 |
| `control_system_builtins.m` | Control System Toolbox: `lqr`, `dlqr`, `place`, `acker`, `care`, `dare`, `lyap`, `dlyap`, `obsv`, `ctrb` with shape rules; `ss`, `tf`, `zpk` recognized | 0 |
| `signal_processing_builtins.m` | Signal Processing Toolbox: `filter`/`filtfilt` passthrough, `conv` symbolic length, window functions, `butter` multi-return | 0 |
| `aerospace_builtins.m` | Aerospace Toolbox: DCM (`angle2dcm` -> 3x3), quaternion (`dcm2quat` -> 1x4, `quatmultiply`), `dcm2angle` multi-return | 0 |
| `core_builtins_dogfood.m` | Dogfood corpus builtins: `bsxfun` broadcast, `interpft` resampling, degree trig, `rmfield`, `nchoosek` | 0 |

>Dimension arithmetic uses canonical polynomial representation to track expressions like `zeros(n+m+1, 2*k)`.

</details>

<details>
<summary><h3>Loops (27 tests)</h3></summary>

Loop analysis with single-pass and fixed-point widening modes (via `--fixpoint`).

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `for_loop_var.m` | Loop variable bound to scalar | 0 |
| `for_range_symbolic.m` | For loop iterates over symbolic range | 0 |
| `nested_loop.m` | Nested for loops handled correctly | 0 |
| `while_basic.m` | While loop with condition | 0 |
| `simple_accumulation.m` | Single-pass mode misses feedback (expected limitation) | 1 |
| `matrix_growth.m` | Matrix growth with symbolic dimensions | 1 |
| `loop_exit_join.m` | Variables unmodified in loop preserve pre-loop state | 0 |
| `loop_may_not_execute.m` | Post-loop join accounts for zero-iteration case | 1 |
| `warning_dedup.m` | Warnings deduplicated inside loops | 1 |
| `fixpoint_convergence.m` | Fixed-point iteration converges in at most 2 iterations | 1 |
| `widen_col_grows.m` | Column grows, row stable: row preserved, column widened | 1 |
| `widen_multiple_vars.m` | Multiple variables with independent stability patterns | 1 |
| `widen_self_reference.m` | Self-referencing updates (`A = A + A`) trigger widening | 1 |
| `widen_while_growth.m` | While loop with matrix growth | 1 |
| `widen_concat_unknown.m` | Concatenation with unknown dimensions in loop | 1 |
| `widen_error_in_branch.m` | Unknown function in conditional branch inside loop | 2 |
| `widen_first_assign_in_body.m` | Variable first assigned inside loop body | 0 |
| `widen_if_in_loop.m` | Conditional growth inside loop converges | 1 |
| `widen_interdependent_vars.m` | Interdependent variables (`A=B`, `B=A+A`) widen correctly | 2 |
| `widen_stable_overwrite.m` | Stable dimension overwrite in loop | 1 |
| `widen_unknown_false_positive.m` | Unknown function doesn't spuriously widen unrelated vars | 1 |
| `widen_unknown_in_body.m` | Unknown function result overwrites variable | 1 |
| `for_iter_count.m` | Iteration count extraction: concrete (`1:5` gives 5) and symbolic (`1:n` gives n) ranges | 2 |
| `for_accum_vertcat.m` | Vertcat accumulation refined: `A=[A;delta]` for 10 iters gives concrete row count | 1 |
| `for_accum_horzcat.m` | Horzcat accumulation refined: `D=[D,delta]` for symbolic `k` iters gives `matrix[5 x (k+2)]` | 1 |
| `for_accum_symbolic.m` | Symbolic range `a:b` accumulation: iteration count `(b-a+1)` used algebraically | 1 |
| `for_accum_no_match.m` | Conservative bailout for self-referencing delta, stepped range, conditional accumulation | 0 |

>Principled widening-based loop analysis uses a 3-phase algorithm (discover, stabilize, post-loop join) that guarantees convergence in at most 2 iterations by widening conflicting dimensions to `None` while preserving stable dimensions.

</details>

<details>
<summary><h3>Functions (74 tests)</h3></summary>

Interprocedural analysis for user-defined functions, anonymous functions (lambdas), nested functions, and workspace-aware external function resolution.

Named Functions (28 tests)

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `simple_function.m` | Basic single-return function analysis | 0 |
| `multiple_returns.m` | Multi-return destructuring assignment | 0 |
| `matrix_constructor.m` | Function with dimension arguments | 0 |
| `procedure.m` | Procedure (no return values) | 1 |
| `unknown_in_function.m` | Unknown shapes propagate from callee to caller | 1 |
| `function_then_script.m` | Function definitions before script statements | 0 |
| `call_with_mismatch.m` | Call with incompatible argument shape | 1 |
| `recursion.m` | Recursive function detected (guard prevents infinite loop) | 1 |
| `cache_hit.m` | Polymorphic cache hit (same args reuse result) | 0 |
| `cache_miss.m` | Polymorphic cache miss (different args re-analyze) | 1 |
| `cache_hit_with_warning.m` | Warnings replayed on cache hit | 2 |
| `cache_symbolic_args.m` | Polymorphic caching with symbolic dimension arguments | 0 |
| `cache_warning_replay.m` | Warning replay preserves both call-site and body line numbers | 2 |
| `return_statement.m` | Return statement exits function early | 0 |
| `return_in_script.m` | Return in script context is an error | 1 |
| `return_in_if.m` | Return inside if-branch (non-returned branch used for join) | 0 |
| `return_in_loop.m` | Return inside loop exits function immediately | 0 |
| `early_return_multi_output.m` | Early return with multiple output variables | 0 |
| `function_in_loop.m` | Function call inside loop body | 0 |
| `nested_function_calls.m` | Nested function calls (`f(g(x))`) | 0 |
| `procedure_with_return.m` | Procedure with explicit return statement | 1 |
| `arg_count_mismatch_cached.m` | Argument count mismatch detected | 1 |
| `endless_basic.m` | Pre-2016 end-less function definition (no `end` keyword) | 0 |
| `endless_inner_blocks.m` | End-less function with nested if/for blocks | 0 |
| `endless_multi.m` | Multiple end-less functions in a single file | 0 |
| `noarg_basic.m` | No-arg procedure syntax `function name` with no parentheses | 0 |
| `space_multi_return.m` | Space-separated multi-return `function [a b c] = f(...)` | 0 |
| `struct_multi_return.m` | Multi-return destructuring into struct fields `[s.x, s.y] = f()` populates the field map correctly | 1 |

Anonymous Functions / Lambdas (17 tests)

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `lambda_basic.m` | Anonymous function definition `@(x) expr` | 0 |
| `lambda_zero_args.m` | Zero-argument lambda definition | 0 |
| `lambda_call_approximate.m` | Lambda body analyzed at call site | 0 |
| `function_handle_from_name.m` | Named function handle `@myFunc` dispatch | 0 |
| `function_handle_join.m` | Function handles join in control flow | 0 |
| `lambda_store_retrieve.m` | Lambda storage with distinct IDs | 0 |
| `lambda_closure_capture.m` | Closure captures environment by-value (MATLAB semantics) | 0 |
| `lambda_call_basic.m` | Lambda body analysis infers result shape from arguments | 0 |
| `lambda_call_closure.m` | Lambda uses closure variable for computation | 0 |
| `lambda_polymorphic_cache.m` | Same lambda called with different arg shapes re-analyzed | 0 |
| `lambda_recursive.m` | Self-referencing lambda caught by recursion guard | 1 |
| `lambda_arg_count_mismatch.m` | Lambda argument count mismatch flagged | 1 |
| `lambda_dim_aliasing.m` | Dimension aliasing through lambda boundaries | 0 |
| `lambda_zero_args_call.m` | Zero-argument lambda call | 0 |
| `lambda_control_flow_join.m` | Different lambdas in if/else both analyzed, results joined | 0 |
| `handle_dispatch_builtin.m` | Function handle `@sin` dispatches to builtin | 0 |
| `handle_dispatch_user_func.m` | Function handle dispatches to user-defined function | 0 |

Workspace Awareness (17 tests)

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `workspace_basic.m` | External function in sibling file resolves silently (no W_UNKNOWN_FUNCTION) | 0 |
| `workspace_handle.m` | Function handle to workspace function works correctly | 0 |
| `workspace_multi_return.m` | Multi-return external functions return unknown for all outputs | 0 |
| `workspace_builtin_priority.m` | Builtins take priority over workspace functions | 0 |
| `workspace_same_file_priority.m` | Same-file functions take priority over workspace functions | 0 |
| `workspace_dim_aliasing.m` | Dimension aliasing across file boundaries (symbolic dims propagate) | 0 |
| `workspace_return.m` | External function with return statement infers correct shape | 0 |
| `workspace_subfunctions.m` | Subfunctions inside external file work correctly | 0 |
| `workspace_cycle_test.m` | Cross-file cycle A->B->A detected; returns unknown gracefully | 0 |
| `workspace_parse_error.m` | Unparseable external file handled gracefully (no caller-visible warning) | 0 |
| `workspace_helper.m` | Helper file for workspace tests (single-return function) | - |
| `workspace_multi_helper.m` | Helper file for workspace tests (multi-return function) | - |
| `workspace_return_helper.m` | Helper file: function with return statement | - |
| `workspace_subfunctions_helper.m` | Helper file: external file with subfunctions | - |
| `workspace_cycle_a.m` | Helper file for cycle test (calls cycle_b) | - |
| `workspace_cycle_b.m` | Helper file for cycle test (calls cycle_a) | - |
| `workspace_parse_error_helper.m` | Helper file with recoverable parse error | - |

Nested Functions (7 tests)

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `nested_def_basic.m` | Nested `function...end` defined inside outer function, callable from within | 0 |
| `nested_def_closure.m` | Nested function reads parent workspace variables | 0 |
| `nested_def_writeback.m` | Nested function writes back to parent workspace variable | 1 |
| `nested_def_dim_alias.m` | Dimension aliasing through nested function boundaries | 0 |
| `nested_def_sibling.m` | Sibling nested functions can call each other via forward references | 0 |
| `nested_def_param_shadow.m` | Nested function parameters shadow parent variables without write-back | 0 |
| `nested_def_scope.m` | Nested function is not visible at top-level script scope | 1 |

Conformal analyzes functions at each call site with the caller's argument shapes, and results are cached per argument shape tuple so the same function called with the same shapes isn't re-analyzed. Symbolic dimension names can propagate across function boundaries, so `f(n)` where `f = @(k) zeros(k,k)` infers `matrix[n x n]`. Lambdas capture their environment by-value at definition time, matching MATLAB semantics. When branches assign different lambdas, Conformal analyzes both bodies and joins the results at the call site.

Nested functions have read access to the parent scope via scope chains, can write back to the parent workspace after returning, and can call sibling nested functions via forward references. A nested function's parameters shadow parent variables without any write-back. Variable lookup walks up the scope chain until it finds the name, so inner functions can read anything defined in any enclosing scope. Multi-level nesting works too, so a function nested three levels deep can read variables from any enclosing level.

External `.m` files are fully parsed and analyzed to infer real return shapes, with cross-file cycles (A->B->A) handled gracefully. Local functions defined inside external files are accessible during cross-file analysis.

</details>

<details>
<summary><h3>Structs (9 tests)</h3></summary>

Struct creation, field access, control-flow joins, and open struct lattice behavior.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `struct_create_assign.m` | Struct creation via field assignment (`s.x = 1`) | 0 |
| `struct_field_access.m` | Field access returns field's shape | 0 |
| `struct_field_not_found.m` | Missing field access on a closed struct emits warning | 1 |
| `struct_field_reassign.m` | Field reassignment with different shape updates field map | 0 |
| `struct_in_control_flow.m` | Struct shape join takes union of fields from both branches | 0 |
| `field_access_unknown.m` | Field access on `unknown` base does not warn; empty matrix promotes to struct on assignment | 0 |
| `open_struct.m` | Field assignment on an unknown base creates an open struct (`struct{x: ..., ...}`); untracked field reads return `unknown` silently; closed+open join produces open | 5 |
| `struct_field_index_assign.m` | Indexed assignment into a struct field `s.x(i) = val` preserves the field's shape | 0 |
| `struct_field_cell_assign.m` | Cell element assignment into a struct field `s.c{i} = val` works correctly | 0 |

>Struct join uses union-with-bottom semantics for closed structs; open structs (from unknown bases) use unknown as the default for missing fields. The lattice ordering is: bottom < closed struct < open struct < unknown.

</details>

<details>
<summary><h3>Cells (27 tests)</h3></summary>

Cell array literals, curly-brace indexing, element assignment, and per-element shape tracking.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `cell_literal.m` | Cell array literal syntax `{1, 2; 3, 4}` | 0 |
| `cell_indexing.m` | Curly-brace indexing `C{i,j}` | 0 |
| `cell_assignment.m` | Cell element assignment `C{i} = expr` | 0 |
| `cell_assign_basic.m` | Basic cell element assignment updates cell shape | 0 |
| `cell_assign_2d.m` | 2D cell element assignment | 0 |
| `cell_assign_after_literal.m` | Cell assignment after literal creation | 0 |
| `cell_assign_non_cell.m` | Cell assignment on non-cell variable emits warning | 1 |
| `cell_assign_updates_element.m` | Cell assignment updates per-element tracking | 0 |
| `cell_builtin.m` | `cell(n)` and `cell(m,n)` constructors | 0 |
| `cell_control_flow_element_join.m` | Per-element tracking joins across control flow branches | 0 |
| `cell_in_control_flow.m` | Cell arrays join across branches | 0 |
| `cell_mixed_types.m` | Cells can hold mixed element types | 0 |
| `cell_symbolic_dims.m` | Cell arrays with symbolic dimensions | 0 |
| `cell_transpose.m` | Cell array transpose `C'` | 0 |
| `cell_range_indexing.m` | Range indexing `C{1:3}` on cell arrays | 0 |
| `cell_element_tracking.m` | Per-element shape tracking with literal indexing | 0 |
| `cell_element_2d_indexing.m` | 2D literal indexing `C{i,j}` extracts precise element shape | 0 |
| `cell_dynamic_indexing.m` | Dynamic indexing joins all element shapes | 0 |
| `cell_2d_linear_indexing.m` | Linear indexing on multi-row cells (column-major) | 0 |
| `cell_end_keyword.m` | `end` keyword in cell indexing `C{end}` | 0 |
| `cell_end_2d.m` | `end` keyword in 2D cell indexing `C{end, end}` | 0 |
| `cell_end_range.m` | `end` as range endpoint `C{1:end}` | 0 |
| `end_arithmetic.m` | `end` keyword arithmetic in cell indexing (`end-1`, `end-2`) | 0 |
| `end_outside_indexing.m` | `end` keyword outside indexing emits warning | 1 |
| `cell_end_assign.m` | `end` keyword in cell assignment LHS (`c{end+1} = val` append pattern) | 0 |
| `curly_indexing_non_cell.m` | Curly indexing on non-cell value is an error | 1 |
| `empty_matrix_promotion.m` | `x = []; x{1} = val` promotes `[]` to cell without warning (MATLAB's universal empty initializer) | 0 |

>Cell arrays use abstract shape `cell[r x c]` with optional per-element tracking. Literal indexing `C{i}` extracts precise element shapes when available. Dynamic indexing joins all elements conservatively. When the index variable has a known concrete value from interval analysis, Conformal can also extract the precise element shape even for variable-index reads. The `end` keyword resolves to the last element index and supports arithmetic (`end-1`, `end/2`).

</details>

<details>
<summary><h3>Recovery (18 tests)</h3></summary>

Parser error recovery and unsupported construct handling (graceful degradation).

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `struct_field.m` | Field access on non-struct value flagged, analysis continues | 1 |
| `cell_array.m` | Curly indexing on non-cell value flagged, analysis continues | 1 |
| `multiple_assignment.m` | Unsupported multiple assignment syntax flagged | 1 |
| `multiline_braces.m` | Multiline cell indexing triggers unsupported construct warning | 1 |
| `dot_elementwise.m` | Dot-elementwise edge cases handled | 0 |
| `end_in_parens.m` | `end` keyword inside parentheses unsupported | 1 |
| `power_recovery.m` | `^` in complex and nested expressions doesn't break recovery | 0 |
| `line_continuation.m` | `...` line continuation in expressions, function args, and matrix literals | 0 |
| `tilde_param.m` | Tilde `~` as unused parameter placeholder in function definitions (`function f(~, data)`) | 0 |
| `dqstring_escape.m` | Double-quoted string literals with `""` escape sequences | 0 |
| `void_return.m` | `function [] = name(x)` void-return function syntax | 0 |
| `bracket_string_concat.m` | Transpose-vs-string disambiguation inside `[]` (space before `'` means string, not transpose) | 0 |
| `parfor_loop.m` | `parfor` loops parsed and analyzed identically to `for` loops | 0 |
| `global_decl.m` | `global` and `persistent` declarations parsed; declared variables treated as `unknown` | 0 |
| `dynamic_field.m` | Dynamic field access `s.(expr)` parses correctly and evaluates to `unknown` | 0 |
| `classdef_suppress.m` | `classdef` blocks consumed without spurious `W_END_OUTSIDE_INDEXING` | 0 |
| `chained_index_struct.m` | Chained indexed struct assignment `A(i).field = val` parses without triggering recovery | 0 |
| `chained_struct_index.m` | Chained struct-index-struct assignment `s.field(i).sub = val` parsed correctly via speculative backtrack | 0 |

>Best-effort analysis. When the parser encounters unsupported syntax, it emits a `W_UNSUPPORTED_*` warning, treats the result as `unknown`, and keeps going.

</details>

<details>
<summary><h3>Constraints (13 tests)</h3></summary>

Dimension constraint solving: equality constraints recorded during operations, validated on concrete bindings, and joined path-sensitively across control flow.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `no_conflict.m` | Compatible symbolic dimensions, no constraint violation | 0 |
| `concrete_no_constraint.m` | Concrete dimensions skip constraint recording (already handled) | 0 |
| `matmul_conflict.m` | Inner dimension constraint conflict detected via `W_CONSTRAINT_CONFLICT` | 1 |
| `elementwise_conflict.m` | Elementwise op constraint conflict detected | 1 |
| `horizontal_concat_conflict.m` | Horizontal concatenation row-count constraint conflict | 1 |
| `vertical_concat_conflict.m` | Vertical concatenation column-count constraint conflict | 1 |
| `symbolic_concrete_constraint.m` | Symbolic dimension constrained by concrete binding | 1 |
| `multi_constraint.m` | Multiple constraints accumulate; first conflict reported | 1 |
| `path_sensitive_join.m` | Constraint added in all branches is kept after join | 0 |
| `path_sensitive_discard.m` | Constraint added in only one branch is discarded after join | 0 |
| `elseif_path_sensitive.m` | Path-sensitive join across elseif chains | 0 |
| `prebound_dim_name.m` | Pre-bound variable names are excluded from constraint recording | 0 |
| `function_scope_isolation.m` | Constraints are scoped to functions and don't leak to callers | 0 |

>Constraint solving operates on `SymDim` polynomial dimensions. When a concrete value is bound to a variable, recorded equality constraints are checked for conflicts and `W_CONSTRAINT_CONFLICT` is emitted. Path-sensitive joins keep only constraints that hold in all branches.

</details>

<details>
<summary><h3>Intervals (19 tests)</h3></summary>

Integer interval domain tracking scalar value ranges for division-by-zero, out-of-bounds indexing, and negative-dimension checks. Conditional interval refinement and symbolic interval bounds.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `interval_basic.m` | Basic interval tracking for scalar integer variables | 0 |
| `interval_branch_join.m` | Interval join across if/else branches (conservative widening) | 0 |
| `division_by_zero.m` | Division by a scalar known to be zero emits `W_DIVISION_BY_ZERO` | 1 |
| `index_out_of_bounds.m` | Index provably outside matrix dimension emits `W_INDEX_OUT_OF_BOUNDS` | 1 |
| `index_in_bounds.m` | Index provably within bounds: no warning | 0 |
| `for_loop_interval.m` | For-loop variable bound to range interval `[1, n]` | 0 |
| `for_loop_index_bounds.m` | Loop variable used as index: bounds checked against dimension | 1 |
| `negative_dim.m` | Provably non-positive dimension expression emits `W_POSSIBLY_NEGATIVE_DIM` | 1 |
| `dim_from_binop.m` | Interval arithmetic in dimension expressions (e.g., `n-1`) | 0 |
| `conditional_refine_basic.m` | `if x > 0` narrows `x` interval in true branch; no false OOB warning | 0 |
| `conditional_refine_compound.m` | Compound `&&` conditions apply both refinements simultaneously | 0 |
| `conditional_refine_eliminates_warning.m` | Guard condition proves index safety, eliminating false-positive OOB | 0 |
| `conditional_refine_else.m` | Condition flipped for else branch (`if x > 3` refines else `x` to `(-inf, 3]`) | 0 |
| `conditional_refine_flipped.m` | Operator flipping: `5 >= x` refines `x` correctly | 0 |
| `conditional_refine_neq.m` | `~=` comparison: no refinement (can't exclude a point from interval) | 0 |
| `conditional_refine_symbolic.m` | Symbolic condition `if n > 0`: refinement with symbolic bounds | 0 |
| `conditional_refine_while.m` | While loop condition refines interval in loop body | 0 |
| `symbolic_interval_for_loop.m` | Symbolic upper bound `for i = 1:n` gives `i` in `[1, n]`; no false OOB on `A(i,:)` | 0 |
| `scalar_propagation.m` | Concrete scalar values propagate into dimension constructors (`m = 3; zeros(m,m)` gives `matrix[3 x 3]`) | 0 |

>Interval analysis runs in parallel with shape inference. `W_INDEX_OUT_OF_BOUNDS` and `W_DIVISION_BY_ZERO` have Error severity (definite runtime errors). Conditional refinement eliminates false positives when branch guards prove safety; symbolic bounds fall back soundly.

</details>

<details>
<summary><h3>Workspace Adversarial (1 test, 20 helpers)</h3></summary>

Adversarial cross-file analysis scenarios: error propagation, struct/cell returns, builtin shadowing, procedure handling, conditional shape joins, subfunctions, accumulation refinement, polymorphic caching stress, and domain-authentic patterns (Kalman, covariance, gradient descent).

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `workspace_mega.m` | Comprehensive cross-file stress test exercising all workspace analysis features | - |
| `ws_accumulate.m` | Helper: for-loop accumulation in external function | - |
| `ws_builtin_chain.m` | Helper: chain of builtin operations across file boundary | - |
| `ws_conditional_shape.m` | Helper: conditional shape join (if/else returning different shapes) | - |
| `ws_covariance.m` | Helper: covariance matrix computation (domain-authentic pattern) | - |
| `ws_gradient_step.m` | Helper: gradient descent step (domain-authentic pattern) | - |
| `ws_kalman_predict.m` | Helper: Kalman filter prediction (domain-authentic pattern) | - |
| `ws_make_cell_pair.m` | Helper: function returning cell array across file boundary | - |
| `ws_make_result.m` | Helper: struct return across file boundary | - |
| `ws_normalize_cols.m` | Helper: column normalization across file boundary | - |
| `ws_procedure_only.m` | Helper: procedure (no return values) across file boundary | - |
| `ws_residual.m` | Helper: residual computation across file boundary | - |
| `ws_return_unknown.m` | Helper: function that returns unknown shape | - |
| `ws_state_update.m` | Helper: state update function across file boundary | - |
| `ws_two_args.m` | Helper: two-argument function across file boundary | - |
| `ws_with_loop.m` | Helper: function with loop body across file boundary | - |
| `ws_with_subfunc.m` | Helper: function with subfunctions across file boundary | - |
| `ws_fill_diag.m` | Helper: function using indexed assignment to fill diagonal; caller infers correct shape | - |
| `sum.m` | Helper: builtin shadowing test (shadows built-in `sum`) | - |
| `ws_continued.m` | Helper: function signature with `...` line continuation across parameters | - |
| `ws_tilde_param.m` | Helper: function with tilde `~` as unused parameter in definition | - |

>These tests exercise cross-file error propagation, struct and cell returns, builtin shadowing, and domain-authentic patterns like Kalman filters and gradient descent.

</details>

<details>
<summary><h3>Workspace (27 tests)</h3></summary>

Cross-file workspace scaling tests: chains, diamond patterns, fan-out/fan-in, polymorphic caching, symbolic dimensions, multi-return, and cycle detection across 26 helper files.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `workspace_scaling.m` | Comprehensive cross-file stress test across all 26 helpers (chains, diamonds, fan-out, linear algebra patterns, cycle detection) | 0 |
| `ws_add_matrices.m` | Helper: element-wise matrix addition across file boundary | - |
| `ws_chain_add.m` | Helper: chained cross-file call (depth 2) | - |
| `ws_compose.m` | Helper: function composition across file boundary | - |
| `ws_deep_chain.m` | Helper: chain of depth 5 across files | - |
| `ws_diamond_left.m` | Helper: left branch of diamond dependency pattern | - |
| `ws_diamond_right.m` | Helper: right branch of diamond dependency pattern | - |
| `ws_diamond_top.m` | Helper: top of diamond (calls left and right) | - |
| `ws_fan_out.m` | Helper: fan-out to multiple sibling files | - |
| `ws_gram.m` | Helper: Gram matrix computation (A'*A) | - |
| `ws_kron_pair.m` | Helper: Kronecker product across file boundary | - |
| `ws_make_rect.m` | Helper: construct rectangular matrix | - |
| `ws_make_sym.m` | Helper: symmetrize a matrix | - |
| `ws_mega_pipeline.m` | Helper: multi-stage pipeline across files | - |
| `ws_normalize.m` | Helper: column normalization | - |
| `ws_outer_product.m` | Helper: outer product computation | - |
| `ws_pipeline.m` | Helper: two-stage pipeline | - |
| `ws_project.m` | Helper: orthogonal projection | - |
| `ws_recursive_a.m` | Helper: cross-file cycle participant (calls recursive_b) | - |
| `ws_recursive_b.m` | Helper: cross-file cycle participant (calls recursive_a) | - |
| `ws_reduce.m` | Helper: reduction to scalar | - |
| `ws_reshape_safe.m` | Helper: safe reshape across file boundary | - |
| `ws_scale.m` | Helper: scalar multiplication across file boundary | - |
| `ws_solve.m` | Helper: linear solve via backslash | - |
| `ws_stack_cols.m` | Helper: horizontal concatenation across file boundary | - |
| `ws_stack_rows.m` | Helper: vertical concatenation across file boundary | - |
| `ws_transform.m` | Helper: affine transformation | - |

>These tests cover the scaling behavior of cross-file analysis, including chains up to depth 5, diamond dependency patterns, and cross-file cycle detection, all with symbolic dimension propagation.

</details>

<details>
<summary><h3>Witness (7 tests)</h3></summary>

Incorrectness witness generation: concrete proofs that dimension conflict warnings are real bugs, not false positives.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `concrete_witness.m` | Trivial witness for concrete dimension mismatch: dims are ints, no symbolic vars needed | 1 |
| `symbolic_witness.m` | Symbolic mismatch (`n` vs `n+1`) is always a conflict; witness enumerates `n` to find a satisfying assignment | 1 |
| `elementwise_witness.m` | Witness for column conflict in element-wise multiplication | 1 |
| `constraint_witness.m` | Witness leverages `scalar_bindings`: `n=3, m=5` are known, so `dim_a=5, dim_b=6` is grounded immediately | 1 |
| `branch_witness.m` | Witness captures active branch path (`n > 3`, true branch) alongside the variable assignments | 1 |
| `no_witness_unknown.m` | Unknown dims produce no warning (dims_definitely_conflict is false), so no witness either | 0 |
| `filter_mode.m` | Confirmed inner-dim mismatch that `--witness filter` would include; tests that witness generation fires correctly | 1 |

>Witness generation runs automatically in the LSP server (enriching hover and diagnostic messages) and can be enabled on the CLI with `--witness`. The `filter` mode produces zero false positives by only surfacing warnings that have a verified concrete counterexample.

</details>

---

### Running the Tests

```bash
# Run all 354 .m tests
cd src && dotnet run -- --tests

# Run with fixed-point loop analysis
cd src && dotnet run -- --fixpoint --tests

# Run strict mode (show all warnings including informational and low-confidence diagnostics)
cd src && dotnet run -- --strict --tests
```

## IDE Integration

The VS Code extension runs the analyzer in-process, since the F# codebase is compiled to JavaScript, using the Fable tool. There is no external runtime dependency: no Python, no .NET, no subprocess. The compiled analyzer is bundled directly into the extension at 181KB.

Firstly, diagnostics appear as underlines as you type, with a configurable 500ms debounce. You can hover any variable to see its inferred shape, including function signatures for user-defined and external functions. Go-to-definition works for user-defined and cross-file functions. If you use `*` where `.*` was probably intended, or `&&` where `&` should be, Conformal suggests the fix. Function definitions show in the sidebar via document symbols, and the status bar tracks warning and error counts along with active modes.

Second, when you save a `.m` file, the server re-analyzes all open files in the same directory, since they could depend on each other. If the server crashes, it auto-restarts up to 3 times. `W_UNSUPPORTED_*` diagnostics render as faded text so they're visually distinct from real errors, and diagnostics with a conflict site link to the original line. Parse errors show as diagnostics rather than crashing the analysis.

Lastly, the extension includes built-in MATLAB syntax highlighting, so you don't need the MathWorks extension.

Configuration settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.fixpoint` | `false` | Enable fixed-point loop analysis (iterative convergence) |
| `conformal.strict` | `false` | Show all warnings including informational and low-confidence diagnostics |
| `conformal.analyzeOnChange` | `true` | Analyze as you type (500ms debounce) |

For editors that can launch a .NET process directly, you can also run the native LSP server:
```bash
cd src && dotnet run -- --lsp
```

## CLI Options

Run `cd src && dotnet run -- file.m` to analyze a file. The other flags are:

`--tests` runs the full test suite.

`--strict` shows all warnings including informational and low-confidence diagnostics.

`--fixpoint` uses fixed-point iteration for loop analysis.

`--witness [MODE]` attaches incorrectness witnesses to dimension conflict warnings. MODE can be `enrich` (the default, prints the witness below each warning), `filter` (only shows warnings with a confirmed witness), or `tag` (prefixes each warning with `[confirmed]` or `[possible]`). The LSP server always runs witness generation and enriches diagnostics automatically.

`--lsp` starts the native .NET Language Server Protocol server.

The exit code is `0` on success and `1` on a parse error, analyzer mismatch, or test failure.

## Real-World Compatibility

To check how Conformal holds up on real MATLAB code, a corpus of 139 `.m` files was drawn from 8 open-source repos on GitHub, covering robotics, signal processing, and scientific computing. In default mode, the corpus produces zero warnings, and in strict mode the warnings are predominantly informational or low-confidence diagnostics.

The repos in the corpus include petercorke/robotics-toolbox-matlab (1491 stars), rpng/kalibr_allan (648 stars), gpeyre/matlab-toolboxes (344 stars), and ImperialCollegeLondon/sap-voicebox (248 stars), among others. These files use a wide range of MATLAB idioms: pre-2016 end-less function definitions, space-separated multi-return syntax (`function [a b c] = f(...)`), Latin-1 encoded files from European authors, `\` for linear solves, and complex matrix literal spacing like `[1 -2; 3 -4]`. Parser robustness improvements came directly from failures on this corpus.

## Limitations

Conformal analyzes a subset of MATLAB. Here's what it doesn't cover:

| Category | What's missing |
|----------|---------------|
| Scope | Workspace analysis covers sibling `.m` files in the same directory. No `addpath` handling or cross-directory resolution yet. |
| Functions | No `varargin`/`varargout`. No `eval`, `feval`, or `str2func`. Nested functions are supported (read/write parent scope, sibling calls, forward references). |
| Builtins | 635 builtins recognized (including Control System, Signal Processing, Aerospace, Optimization, Mapping, Image Processing, Robotics, Statistics, Communications, Computer Vision, Deep Learning, and Symbolic Math Toolbox functions); 315 have explicit shape rules. Unrecognized calls produce a `W_UNKNOWN_FUNCTION` warning (strict-only by default). |
| Cell arrays | Per-element tracking available for literal-indexed cells. Dynamic indexing conservatively joins all elements. |
| Indexing | `end` keyword supported with arithmetic (`C{end}`, `C{end-1}`, `A(1:end)`, `A(end-2:end, :)`). Variable operands in `end` arithmetic fall through to conservative join. |
| Data types | No classes, no maps, no tables, no N-D arrays (only 2-D matrices). No complex number tracking. |
| Syntax | No command-style calls (`save file.mat`). `global`/`persistent` declarations are parsed (targets get `unknown` shape). `parfor` is treated as a regular `for` loop. `classdef` blocks are parsed and suppressed without side effects. |
| I/O and graphics | `load`, `save`, `fprintf`, `plot`, and other side-effecting functions are recognized (no spurious `W_UNKNOWN_FUNCTION`) but their return shapes are not tracked. |
| Dynamic features | No `eval`, no `feval`, no `str2func`. Dynamic field access `s.(expr)` is parsed and evaluates to `unknown`. Runtime type introspection beyond type predicates (`iscell`, `isscalar`, `isnumeric`, etc.) is not tracked. |

These are deliberate scope boundaries, not bugs. Conformal focuses on the matrix-heavy computational core of MATLAB, where dimension errors are most common and most costly.

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

### Roadmap

First, the immediate priorities are propagating constraints across function boundaries and improving the GitHub Action for CI integration. Further out, I'd like to support additional editors like Neovim, add cross-directory workspace analysis with `addpath` handling, and explore integration with MATLAB's built-in Code Analyzer.
