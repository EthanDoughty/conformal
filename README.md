<div align="center">

# Conformal

### Static Shape & Dimension Analysis for MATLAB

[![Version](https://img.shields.io/badge/version-0.14.0-orange.svg)](#motivation-and-future-directions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen.svg)](#test-suite)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

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
$ python3 conformal.py example.m

Warnings:
  - Line 3: Dimension mismatch in expression (A * B):
    inner dims 4 vs 5 (shapes matrix[3 x 4] and matrix[5 x 2])
  - Line 4: Vertical concatenation requires equal column counts
    across rows; got 4 and 2 in matrix literal.

Final environment:
    Env{A: matrix[3 x 4], B: matrix[5 x 2], C: unknown, D: unknown}
```

## Requirements

- Python 3.10+
- No third-party dependencies
- Tested on Linux
- No MATLAB installation required

## What the Analysis Detects

All warnings include source line numbers. When the analyzer finds a definite error, it marks the result as `unknown` and keeps going so you get as many diagnostics as possible in a single pass.

### Operations

Dimension mismatches in `+`, `-`, `*`, `.*`, `./`. Inner dimension checks for matrix multiplication. Scalar-matrix broadcasting (`s*A`, `s + A`) is handled correctly. When you use `*` where `.*` was probably intended, the analyzer suggests the fix.

### Literals and concatenation

Matrix literals `[1 2; 3 4]`, cell array literals `{1, 2; 3, 4}`, and string literals (`'hello'`, `"world"`). Horizontal concatenation `[A B]` checks that row counts match; vertical concatenation `[A; B]` checks columns. Symbolic dimensions compose through concatenation, so `[A B]` where A is `n x k` and B is `n x m` gives `n x (k+m)`.

### Indexing

Parenthesized indexing `A(i,j)`, slice indexing `A(:,j)` and `A(i,:)`, range indexing `A(2:5,:)`, linear indexing, and full-matrix `A(:,:)`. Curly-brace indexing `C{i,j}` for cell arrays with per-element shape tracking (literal index extracts precise element shape). Cell element assignment `C{i} = expr`. The `end` keyword works in indexing contexts (`C{end}`, `A(1:end, 2)`).

### Functions

20+ builtins with shape rules: `zeros`, `ones`, `eye`, `rand`, `randn`, `reshape`, `repmat`, `diag`, `det`, `inv`, `linspace`, `norm`, `size`, `length`, `numel`, `transpose`, `horzcat`, `vertcat`, `iscell`, `isscalar`, and more. Dimension arithmetic works inside builtin arguments, so `zeros(n+1, 2*m)` is tracked symbolically.

User-defined functions are analyzed at each call site with the caller's argument shapes. Three forms: single return (`function y = f(x)`), multi-return (`function [a, b] = f(x)`), and procedures (`function f(x)`). Anonymous functions `@(x) expr` are analyzed the same way, with by-value closure capture at definition time. Function handles `@funcName` dispatch to their targets. Results are cached per argument shape tuple so the same function called with the same shapes isn't re-analyzed.

### Data structures

Structs with field assignment (`s.x = A`), field access, and chained dot notation (`s.x.y`). Missing field access emits a warning. Struct shapes join across branches by taking the union of fields.

Cell arrays with `cell(n)` and `cell(m,n)` constructors, curly-brace indexing, and element assignment. Literal indexing `C{i}` extracts the precise shape of element `i` when available. Dynamic indexing joins all element shapes conservatively. Curly-brace indexing on a non-cell emits a warning.

### Control flow

`if`/`elseif`/`else`, `for`, `while`, `switch`/`case`/`otherwise`, `try`/`catch`, `break`, `continue`, `return`. When branches assign different shapes to the same variable, the analyzer joins them conservatively. Loops use a single pass by default, or widening-based fixed-point iteration via `--fixpoint` for guaranteed convergence (≤2 iterations).

### Symbolic dimensions

Variables with unknown concrete size get symbolic names like `n`, `m`, `k`. These propagate through operations: `1:n` gives a `1 x n` vector, `[A B]` computes `n x (k+m)`, `zeros(n+1, 2*m)` tracks the arithmetic. Symbolic dimensions are represented as canonical polynomials with rational coefficients, so `n+m` and `m+n` are recognized as equal, and `n+n` simplifies to `2*n`. When a function `f(k)` is called with a symbolic argument `n`, the dimension name `n` propagates into the function body.

## Language Coverage

The analyzer parses and tracks shapes through:

| Category | Constructs |
|----------|-----------|
| Expressions | `+`, `-`, `*`, `.*`, `./`, `==`, `~=`, `<`, `>`, `<=`, `>=`, `&&`, `\|\|`, `~`, `'` |
| Literals | `[1 2; 3 4]`, `{1, 2; 3, 4}`, `'string'`, `"string"`, `1:n` |
| Indexing | `A(i,j)`, `A(:,j)`, `A(2:5,:)`, `C{i}`, `C{i} = x` |
| Assignment | `x = expr`, `s.field = expr`, `C{i} = expr`, `[a, b] = f(x)` |
| Functions | `function y = f(x)`, `@(x) expr`, `@funcName`, 20+ builtins |
| Control flow | `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`, `try`/`catch` |
| Statements | `break`, `continue`, `return` |
| Data types | scalars, matrices, strings, structs, cell arrays, function handles |

## Shape System

Every expression gets a shape from the abstract domain. There are 7 kinds:

| Shape | Example | Notes |
|-------|---------|-------|
| `scalar` | `5`, `x` | Single numeric value |
| `matrix[r x c]` | `matrix[3 x 4]`, `matrix[n x m]` | Dimensions can be concrete, symbolic, or unknown |
| `string` | `'hello'` | Char array |
| `struct{...}` | `struct{x: scalar, y: matrix[3 x 1]}` | Tracks fields and their shapes |
| `function_handle` | `@(x) x'`, `@sin` | Tracks lambda ID for join precision |
| `cell[r x c]` | `cell[3 x 1]` | Cell array with optional per-element shape tracking |
| `unknown` | | Error or indeterminate; the lattice top |

Dimensions in `matrix[r x c]` can be:
- Concrete integers: `3`, `100`
- Symbolic names: `n`, `m`, `k`
- Symbolic expressions: `n+m`, `2*k`, `n+1`
- Unknown: `None` (no information available)

Symbolic dimensions use a frozen polynomial representation (`SymDim`) with rational coefficients. Canonicalization handles commutativity (`n+m` = `m+n`), like-term collection (`n+n` = `2*n`), and constant-offset conflict detection. When control flow branches assign conflicting dimensions to the same variable, the analyzer joins them to `None` (unknown). In loops with `--fixpoint`, conflicting dimensions get widened to `None` while stable dimensions are preserved.

## Project Structure

```
frontend/    Parsing and IR lowering
ir/          Typed IR dataclass definitions
analysis/    Shape analysis, diagnostics, and core semantics
runtime/     Shape domain and environments
tests/       Self-checking MATLAB programs
tools/       Debugging utilities (AST printer)
```

## Test Suite

The analyzer is validated by 147 self-checking test programs organized into 12 categories. Each test embeds its expected behavior as inline assertions:

```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]   % Override for --fixpoint mode
```

The test runner validates that the analyzer's output matches these expectations, ensuring correctness across all supported language features.

---

<details open>
<summary><h3>Basics (7 tests)</h3></summary>

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

</details>

<details open>
<summary><h3>Symbolic Dimensions (6 tests)</h3></summary>

Tests symbolic dimension tracking, arithmetic, and canonical polynomial representation introduced in v0.13.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `dimension_tracking.m` | Colon vectors preserve symbolic dimensions (`1:n`) | 0 |
| `dimension_arithmetic.m` | Addition/multiplication of symbolic dimensions (e.g., `n+m`, `2*k`) | 0 |
| `canonicalization.m` | SymDim polynomial canonicalization ensures `zeros(n,m)` joins with `zeros(n,m)` | 0 |
| `commutativity_join.m` | Commutative equality: `(n+m)` joins with `(m+n)` | 0 |
| `like_terms.m` | Like-term collection: `(n+n)` canonicalizes to `(2*n)` | 0 |
| `rational_dimensions.m` | Rational coefficients in symbolic dimensions (e.g., `n/2`) | 0 |

>Symbolic dimensions are represented as frozen polynomial dataclasses (`SymDim`) with canonical equality and rational coefficients, enabling precise tracking of parametric shapes across function boundaries.

</details>

<details open>
<summary><h3>Indexing (7 tests)</h3></summary>

MATLAB-style indexing including scalar, slice, range, and linear indexing.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `scalar_index.m` | Scalar indexing `A(i,j)` returns scalar | 0 |
| `slice_index.m` | Slice indexing `A(:,j)` and `A(i,:)` shape rules | 0 |
| `range_index.m` | Range indexing `A(2:5, :)` preserves symbolic dimensions | 0 |
| `linear_index.m` | Linear indexing on scalar base is an error | 1 |
| `invalid_row_index.m` | Constant-range row indexing edge cases | 0 |
| `invalid_col_index.m` | Constant-range column indexing edge cases | 0 |
| `invalid_linear_index.m` | Non-scalar index argument flagged | 1 |

</details>

<details open>
<summary><h3>Control Flow (15 tests)</h3></summary>

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

>Conservative join semantics. When branches disagree on a variable's shape, the analyzer joins to the least upper bound (often `unknown`).

</details>

<details>
<summary><h3>Literals (7 tests)</h3></summary>

Matrix literals, string literals, and concatenation constraints.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `matrix_literal.m` | Basic matrix literals `[1 2; 3 4]` parsed and shaped correctly | 0 |
| `horzcat_vertcat.m` | Horizontal/vertical concatenation dimension constraints | 1 |
| `symbolic_concat.m` | Symbolic dimension addition in concatenation (e.g., `[A B]` → `n x (k+m)`) | 0 |
| `string_literal.m` | String literals with both quote styles (`'foo'`, `"bar"`) | 0 |
| `string_horzcat.m` | String concatenation via horizontal concatenation | 0 |
| `string_matrix_error.m` | String-matrix arithmetic operations flagged | 1 |
| `string_in_control_flow.m` | String/scalar shape joins across branches | 0 |

</details>

<details>
<summary><h3>Builtins (9 tests)</h3></summary>

Shape rules for 20 recognized MATLAB builtins, call/index disambiguation, and dimension arithmetic.

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

>Dimension arithmetic uses canonical polynomial representation to track expressions like `zeros(n+m+1, 2*k)`.

</details>

<details>
<summary><h3>Loops (22 tests)</h3></summary>

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
| `fixpoint_convergence.m` | Fixed-point iteration converges in ≤2 iterations | 1 |
| `widen_col_grows.m` | Column grows, row stable → row preserved, column widened | 1 |
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

>Principled widening-based loop analysis (v0.9.2) uses a 3-phase algorithm (discover, stabilize, post-loop join) that guarantees convergence in ≤2 iterations by widening conflicting dimensions to `None` while preserving stable dimensions.

</details>

<details>
<summary><h3>Functions (39 tests)</h3></summary>

Interprocedural analysis for user-defined functions and anonymous functions (lambdas).

Named Functions (21 tests)

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
| `cache_hit.m` | Polymorphic cache hit (same args → reuse result) | 0 |
| `cache_miss.m` | Polymorphic cache miss (different args → re-analyze) | 1 |
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

Anonymous Functions / Lambdas (18 tests)

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

Key features:
- Interprocedural analysis: functions analyzed at each call site with the caller's argument shapes
- Polymorphic caching: results cached per `(func_name, arg_shapes)` to avoid redundant re-analysis
- Dimension aliasing: symbolic dimension names propagate across boundaries (e.g., `f(n)` where `f = @(k) zeros(k,k)` infers `matrix[n x n]`)
- Lambda closure capture: by-value environment capture at definition time (MATLAB semantics)
- Control flow precision: when branches assign different lambdas, both bodies are analyzed and results joined at call site

</details>

<details>
<summary><h3>Structs (5 tests)</h3></summary>

Struct creation, field access, and control-flow joins.

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `struct_create_assign.m` | Struct creation via field assignment (`s.x = 1`) | 0 |
| `struct_field_access.m` | Field access returns field's shape | 0 |
| `struct_field_not_found.m` | Missing field access emits warning | 1 |
| `struct_field_reassign.m` | Field reassignment with different shape updates field map | 0 |
| `struct_in_control_flow.m` | Struct shape join takes union of fields from both branches | 0 |

>Struct join uses union-with-bottom semantics. Fields present in only one branch get `bottom` in the other, then join to `unknown`.

</details>

<details>
<summary><h3>Cells (24 tests)</h3></summary>

Cell array literals, curly-brace indexing, element assignment, and per-element shape tracking (v0.12.2-0.14.0).

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
| `end_outside_indexing.m` | `end` keyword outside indexing emits warning | 1 |
| `curly_indexing_non_cell.m` | Curly indexing on non-cell value is an error | 1 |

>Cell arrays use abstract shape `cell[r x c]` with optional per-element tracking. Literal indexing `C{i}` extracts precise element shapes when available. Dynamic indexing joins all elements conservatively. The `end` keyword resolves to the last element index.

</details>

<details>
<summary><h3>Recovery (6 tests)</h3></summary>

Parser error recovery and unsupported construct handling (graceful degradation).

| Test | What It Validates | Warnings |
|------|-------------------|----------|
| `struct_field.m` | Field access on non-struct value flagged, analysis continues | 1 |
| `cell_array.m` | Curly indexing on non-cell value flagged, analysis continues | 1 |
| `multiple_assignment.m` | Unsupported multiple assignment syntax flagged | 1 |
| `multiline_braces.m` | Multiline cell indexing triggers unsupported construct warning | 1 |
| `dot_elementwise.m` | Dot-elementwise edge cases handled | 0 |
| `end_in_parens.m` | `end` keyword inside parentheses unsupported | 1 |

>Best-effort analysis. When the parser encounters unsupported syntax, it emits a `W_UNSUPPORTED_*` warning, treats the result as `unknown`, and keeps going.

</details>

---

### Running the Tests

```bash
# Run all 147 tests
make test
python3 conformal.py --tests

# Run with fixed-point loop analysis
python3 conformal.py --fixpoint --tests

# Run strict mode (fail on unsupported constructs)
python3 conformal.py --strict --tests
```

## Getting Started

Clone and Verify:
```bash
git clone https://github.com/EthanDoughty/conformal.git
cd conformal
make test
```

Analyze a file:
```bash
make run FILE=tests/basics/inner_dim_mismatch.m
```

## CLI Options

`conformal.py file.m` – analyze a file (IR-based)

`--tests` – run full test suite

`--strict` – fail on unsupported constructs

`--fixpoint` – use fixed-point iteration for loop analysis

Exit codes:

`0` – success

`1` – parse error, analyzer mismatch, or test failure

## Why Python?

Most of the work in this project is tree manipulation: walking ASTs, matching patterns on IR nodes, joining lattice elements, and tracking dictionaries of variable shapes. Python is well-suited for this. Dataclasses make clean IR nodes, dicts are first-class, and there's no compile step slowing down iteration. The entire analyzer ships as a single codebase with zero dependencies.

The obvious downside is speed. For CLI use and files up to a few hundred lines, it's plenty fast. For IDE integration (the 1.0 goal), latency will matter more. The plan there is incremental analysis and per-function caching, not a full rewrite. The pipeline is cleanly layered (parser → IR → analysis), so if profiling reveals a bottleneck, it can be addressed without rearchitecting.

For adoption, the distribution story matters more than the language. A VS Code extension that bundles everything and just works will get more users than a fast binary that requires manual setup.

## Limitations

Conformal analyzes a subset of MATLAB. Here's what it doesn't cover:

| Category | What's missing |
|----------|---------------|
| Scope | Single-file analysis only. No cross-file function resolution or `addpath` handling. |
| Functions | No nested functions. No `varargin`/`varargout`. No `eval`, `feval`, or `str2func`. |
| Builtins | 21 builtins recognized. Toolbox functions (`fft`, `eig`, `svd`, `conv`, `filter`, ...) are not modeled and produce an unknown-function warning. |
| Cell arrays | Per-element tracking available for literal-indexed cells. Dynamic indexing conservatively joins all elements. |
| Indexing | `end` keyword supported in simple contexts (`C{end}`, `A(1:end)`). Arithmetic with `end` (e.g., `end-1`) not yet supported. |
| Data types | No classes, no maps, no tables, no N-D arrays (only 2-D matrices). No complex number tracking. |
| Syntax | No command-style calls (`save file.mat`), no `global`/`persistent`, no `parfor`, no `classdef`. |
| I/O and graphics | No `load`, `save`, `fprintf`, `plot`, or any side-effecting functions. |
| Dynamic features | No `eval`, no dynamic field access (`s.(name)`), no runtime type introspection beyond `iscell`/`isscalar`. |

These are deliberate scope boundaries, not bugs. The analyzer focuses on the matrix-heavy computational core of MATLAB where dimension errors are most common and most costly.

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

### Roadmap

**Near-term**
- Symbolic constraint solving (builds on the `SymDim` polynomial domain)
- Nested function support
- Per-element cell array tracking
- Expanded builtin coverage (toolbox functions)

**IDE / LSP Integration (1.0)**

The 1.0 goal is getting Conformal into editors. The plan is a hybrid: a Python LSP server ([pygls](https://github.com/openlawlibrary/pygls)) that runs the analyzer as a long-lived process, wrapped in a TypeScript VS Code extension that handles the editor side.

What that looks like in practice:
- Squiggly underlines on dimension mismatches as you type
- Hover tooltips showing inferred shapes (`matrix[3 x n]`)
- The extension bundles everything, so users install one thing from the VS Code marketplace. No Python setup, no pip.

This is the same split architecture that Pylance, ruff, and jedi-language-server use. The Python analysis engine stays as-is; the TypeScript layer is just a thin LSP client.

**Long-term**
- Additional editors (Neovim, MATLAB Online)
- Workspace-level analysis across multiple files
- Integration with MATLAB's built-in Code Analyzer
