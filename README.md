# Mini-MATLAB Static Shape & Dimension Analysis
**Ethan Doughty**

This project implements a static shape and dimension analysis for a carefully chosen subset of the MATLAB programming language, referred to as Mini-MATLAB.

The goal of the analysis is to detect common matrix-related errors before runtime, using a custom parser and static analyzer designed specifically for MATLAB-style matrix semantics. The tool reasons about matrix shapes, symbolic dimensions, and control flow without relying on the MATLAB runtime.

## Requirements

- **Python 3.10+**
- No third-party dependencies
- Tested on Linux
- No MATLAB installation required

## What the Analysis Detects

The analyzer statically detects dimension and shape issues in the following constructs:

- **Matrix addition and subtraction** (`+`, `-`)
- **Matrix multiplication** (`*`)
- **Elementwise operations** (`.*`, `./`, `/`)
- **Scalar–matrix operations** (`s*A`, `s + A`)
- **Matrix literals** (`[1 2; 3 4]`, `[A B]`, `[A; B]`)
- **Horizontal and vertical concatenation constraints**
- **Vector transpose** (`v'`)
- **Colon-generated vectors** (`1:n`)
- **MATLAB-style indexing and slices** (`A(i,j)`, `A(i,:)`, `A(:,j)`, `A(:,:)`)
- **Range indexing** (`A(2:5,:)`, `A(:,2:5)`)
- **Matrix–scalar comparisons** (`A == 0`)
- **Logical operators on non-scalars** (`&&`, `||`)
- **Incompatible variable reassignments**
- **MATLAB-aware fix suggestions**
  - e.g. suggesting `.*` instead of `*`

All warnings are reported with source line numbers, and the analysis continues in a "best-effort" manner even after detecting errors.

## Language Subset Design

The language subset and analysis design were chosen to isolate a fragment of MATLAB that is dense enough to show interesting behaviors, but small enough to analyze with a custom static tool.

The subset includes:

- assignments and expressions
- function calls (19 recognized builtins with full shape rule coverage)
- user-defined functions (single/multi-return/procedure forms with interprocedural analysis)
- anonymous functions (lambda body analysis at call sites with closure capture)
- function handles (named handles dispatch to their targets)
- control flow (if/elseif/else, for, while, switch/case, try/catch, break, continue, return)
- symbolic dimensions
- indexing and transpose
- strings (char array literals with MATLAB-faithful arithmetic)
- structs (field access and assignment with chained dot notation)

Loops are analyzed using a single pass by default, or with principled widening via `--fixpoint` for guaranteed convergence. The widening-based analysis accelerates convergence (≤2 iterations) by widening conflicting dimensions to unknown while preserving stable dimensions.

## Shape System

Each expression is assigned a shape from a small abstract domain:

- `scalar`
- `matrix[r x c]` where `r` and `c` may be:
  - concrete integers
  - symbolic names (`n`, `m`, `k`)
  - unknown (`None`)
- `string` (char array literals)
- `struct{fields}` (struct values with named fields)
- `function_handle` (anonymous functions and named handles)
- `unknown`

The analysis supports:
- symbolic dimension equality
- symbolic dimension joins across control flow
- symbolic dimension addition for matrix concatenation (e.g. `n x (k+m)`)
- symbolic dimension multiplication for replication (e.g. `(n*k)`)
- symbolic dimension arithmetic in builtin arguments (e.g. `zeros(n+1, m)`)
- widening for loop convergence (stable dims preserved, conflicting dims → None)
- dimension aliasing across function boundaries (caller's symbolic names propagate to callee)
- lambda body analysis at call sites (polymorphic caching per argument shapes)
- closure capture for anonymous functions (by-value environment capture at definition)

## Project Structure

frontend/    Parsing and IR lowering
ir/          Typed IR dataclass definitions
analysis/    Shape analysis, diagnostics, and core semantics
legacy/      Original syntax-based analyzer (for comparison)
runtime/     Shape domain and environments
tests/       Self-checking Mini-MATLAB programs
tools/       Debugging utilities (AST printer)

## Test Suite

The project includes a self-checking test suite of Mini-MATLAB programs.

Each test file:
- documents its intent using MATLAB comments
- declares expected warnings and final shapes using inline assertions:
  ```matlab
  % EXPECT: warnings = 1
  % EXPECT: A = matrix[n x (k+m)]
  ```

| Tests | Category
|------|------------
| basics/valid_add.m | Basic addition
| basics/invalid_add.m | Invalid addition (dimension mismatch)
| basics/matrix_multiply.m | Matrix multiplication
| basics/inner_dim_mismatch.m | Inner dimension mismatch
| basics/scalar_matrix_ops.m | Scalar-matrix operations
| basics/elementwise_ops.m | Elementwise operations with errors
| basics/reassignment.m | Variable reassignment
| symbolic/dimension_tracking.m | Colon vectors and symbolic dimensions
| symbolic/dimension_arithmetic.m | Symbolic dimension arithmetic
| indexing/scalar_index.m | Scalar indexing
| indexing/slice_index.m | Slice indexing
| indexing/range_index.m | Range indexing
| indexing/linear_index.m | Linear indexing (errors)
| indexing/invalid_row_index.m | Invalid row index
| indexing/invalid_col_index.m | Invalid column index
| indexing/invalid_linear_index.m | Invalid linear index
| control_flow/if_branch_mismatch.m | Control-flow joins with mismatches
| control_flow/suspicious_comparison.m | Matrix-scalar comparisons
| control_flow/if_else_error_branch.m | If-else with error in branch
| control_flow/elseif_chain.m | Elseif chains
| control_flow/elseif_no_else.m | Elseif without else
| control_flow/elseif_mismatch.m | Elseif branch mismatches
| control_flow/break_simple.m | Break statement in loop
| control_flow/continue_simple.m | Continue statement in loop
| control_flow/break_nested_loop.m | Break in nested loops
| control_flow/switch_basic.m | Switch/case statement
| control_flow/switch_no_otherwise.m | Switch without otherwise
| control_flow/switch_mismatch.m | Switch case type mismatches
| control_flow/try_catch_basic.m | Try/catch exception handling
| control_flow/try_catch_no_error.m | Try/catch with no error
| control_flow/try_nested.m | Nested try/catch
| literals/matrix_literal.m | Matrix literals
| literals/horzcat_vertcat.m | Horizontal and vertical concatenation
| literals/symbolic_concat.m | Symbolic concatenation
| literals/string_literal.m | String literals (char arrays)
| literals/string_horzcat.m | String concatenation via horzcat
| literals/string_matrix_error.m | Invalid string arithmetic warning
| literals/string_in_control_flow.m | String/scalar join in control flow
| builtins/unknown_function.m | Unknown function warning
| builtins/shape_preserving.m | Shape-preserving builtins
| builtins/call_vs_index.m | Call vs index disambiguation
| builtins/constructors.m | Matrix constructors and element-wise builtins
| builtins/remaining_builtins.m | Complete builtin coverage (det/diag/inv/linspace/norm)
| functions/simple_function.m | Basic user-defined function
| functions/multiple_returns.m | Multi-return function with destructuring (+ single-output usage)
| functions/matrix_constructor.m | Function returning symbolic-shaped matrix
| functions/procedure.m | Procedure (no return values)
| functions/unknown_in_function.m | Unknown shapes in function bodies
| functions/function_then_script.m | Function definitions followed by script
| functions/call_with_mismatch.m | Function call with dimension mismatch
| functions/recursion.m | Recursive function calls
| functions/cache_hit.m | Polymorphic cache hit (same arg shapes)
| functions/cache_miss.m | Polymorphic cache miss (different arg shapes)
| functions/cache_hit_with_warning.m | Warning replay on cache hit
| functions/return_statement.m | Return statement (early exit + unreachable code)
| functions/return_in_script.m | Return in script context (warning)
| functions/return_in_if.m | Return statement inside if-branch
| functions/return_in_loop.m | Return statement inside loop body
| functions/cache_symbolic_args.m | Cache with symbolic dimension arguments
| functions/cache_warning_replay.m | Warning replay on cache hit
| functions/early_return_multi_output.m | Early return in multi-output function
| functions/function_in_loop.m | Function call inside loop body
| functions/nested_function_calls.m | Nested user-defined function calls
| functions/procedure_with_return.m | Procedure with explicit return
| functions/arg_count_mismatch_cached.m | Arg count mismatch (no cache interaction)
| functions/lambda_basic.m | Anonymous function definition and assignment
| functions/lambda_call_approximate.m | Lambda call with body analysis (no longer approximate)
| functions/lambda_zero_args.m | Zero-argument lambda functions
| functions/function_handle_from_name.m | Named function handles (`@myFunc`)
| functions/function_handle_join.m | Function handle join in control flow
| functions/lambda_store_retrieve.m | Lambda storage with distinct IDs
| functions/lambda_closure_capture.m | Closure captures environment by-value
| functions/lambda_call_basic.m | Lambda body analysis with shape inference
| functions/lambda_call_closure.m | Lambda uses closure variable in body
| functions/lambda_polymorphic_cache.m | Same lambda, different arg shapes (cache)
| functions/lambda_recursive.m | Self-referencing lambda (recursion guard)
| functions/lambda_arg_count_mismatch.m | Lambda argument count mismatch
| functions/lambda_dim_aliasing.m | Dimension aliasing through lambda calls
| functions/lambda_zero_args_call.m | Zero-argument lambda call
| functions/lambda_control_flow_join.m | Lambda in if/else branches, joined call
| functions/handle_dispatch_builtin.m | Named handle dispatches to builtin
| functions/handle_dispatch_user_func.m | Named handle dispatches to user function
| structs/struct_create_assign.m | Struct creation and field assignment
| structs/struct_field_access.m | Chained struct field access (`s.a.b`)
| structs/struct_field_not_found.m | Missing struct field warning
| structs/struct_in_control_flow.m | Struct join across branches
| structs/struct_field_reassign.m | Struct field reassignment
| recovery/struct_field.m | Field access on non-struct value (promoted from recovery)
| recovery/cell_array.m | Unsupported cell array indexing
| recovery/multiple_assignment.m | Unsupported multiple assignment
| recovery/multiline_braces.m | Unsupported multiline cell indexing
| recovery/dot_elementwise.m | Dot-elementwise edge cases
| recovery/end_in_parens.m | Unsupported end in parentheses

## Getting Started

Clone and Verify
`git clone <repo-url>`
`cd matlab-static-dimension-analysis`
`make test`

Analyze a File
`make run FILE=tests/basics/inner_dim_mismatch.m`

Example Output
`Warnings:`
  `- Line 11: Dimension mismatch in expression (A * x):`
    `inner dims 4 vs 5 (shapes matrix[3 x 4] and matrix[5 x 1])`

`Final environment:`
`Env{A: matrix[3 x 4], x: matrix[5 x 1], y: unknown}`

## CLI Options

mmshape.py file.m – analyze a file (IR-based)

`--tests` – run full test suite

`--strict` – fail on unsupported constructs

`--fixpoint` – use fixed-point iteration for loop analysis

Exit codes:

0 – success

1 – parse error, analyzer mismatch, or test failure

## Notes and Challenges

- The IR enforces structural invariants absent in the raw syntax AST

- Matrix literals are parsed with MATLAB-aware rules.

- The analyzer uses best-effort inference; Even when a definite mismatch is detected, it continues analysis to provide as much information as possible.

- The analyzer is strict on provable dimension errors. When an operation is definitely invalid (e.g., inner-dimension mismatch in A*B), it emits a warning and treats the expression result as unknown.

## Limitations
This tool does not support:
- cell arrays
- file I/O
- plotting or graphics
- precise loop invariants
- nested functions

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

Possible future extensions include:

- nested functions
- stricter invalidation semantics for definite errors
- richer symbolic constraint solving
- IDE or language-server integration
