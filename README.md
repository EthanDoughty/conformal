# MATLAB Static Shape & Dimension Analysis
**Ethan Doughty**

This project implements a static shape and dimension analysis for MATLAB code.

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
- **Cell array literals** (`{1, 2; 3, 4}`)
- **Horizontal and vertical concatenation constraints**
- **Vector transpose** (`v'`)
- **Colon-generated vectors** (`1:n`)
- **MATLAB-style indexing and slices** (`A(i,j)`, `A(i,:)`, `A(:,j)`, `A(:,:)`)
- **Curly-brace indexing** (`C{i,j}`)
- **Range indexing** (`A(2:5,:)`, `A(:,2:5)`)
- **Matrix–scalar comparisons** (`A == 0`)
- **Logical operators on non-scalars** (`&&`, `||`)
- **Incompatible variable reassignments**
- **MATLAB-aware fix suggestions**
  - e.g. suggesting `.*` instead of `*`

All warnings are reported with source line numbers, and the analysis continues in a "best-effort" manner even after detecting errors.

## Language Coverage

The analyzer supports MATLAB code with the following features:

- Assignments and expressions
- Function calls (20 recognized builtins with full shape rule coverage)
- User-defined functions (single/multi-return/procedure forms with interprocedural analysis)
- Anonymous functions (lambda body analysis at call sites with closure capture)
- Function handles (named handles dispatch to their targets)
- Control flow (if/elseif/else, for, while, switch/case, try/catch, break, continue, return)
- Symbolic dimensions
- Indexing and transpose
- String literals (char array literals with MATLAB-faithful arithmetic)
- Structs (field access and assignment with chained dot notation)
- Cell arrays (literals, curly indexing, `cell()` builtin)

Loops are analyzed using a single pass by default, or with principled widening via `--fixpoint` for guaranteed convergence. The widening-based analysis accelerates convergence (≤2 iterations) by widening conflicting dimensions to unknown while preserving stable dimensions.

## Shape System

Each expression is assigned a shape from an abstract domain:

- `scalar`
- `matrix[r x c]` where `r` and `c` may be:
  - concrete integers
  - symbolic names (`n`, `m`, `k`)
  - unknown (`None`)
- `string` (char array literals)
- `struct{fields}` (struct values with named fields)
- `function_handle` (anonymous functions and named handles)
- `cell[r x c]` (cell array with given dimensions)
- `unknown` (error or indeterminate shape)

The analysis supports:
- Symbolic dimension equality
- Symbolic dimension joins across control flow
- Symbolic dimension addition for matrix concatenation (e.g. `n x (k+m)`)
- Symbolic dimension multiplication for replication (e.g. `(n*k)`)
- Symbolic dimension arithmetic in builtin arguments (e.g. `zeros(n+1, m)`)
- Widening for loop convergence (stable dims preserved, conflicting dims → None)
- Dimension aliasing across function boundaries (caller's symbolic names propagate to callee)
- Lambda body analysis at call sites (polymorphic caching per argument shapes)
- Closure capture for anonymous functions (by-value environment capture at definition)

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

The project includes a self-checking test suite of MATLAB programs.

Each test file:
- Documents its intent using MATLAB comments
- Declares expected warnings and final shapes using inline assertions:
  ```matlab
  % EXPECT: warnings = 1
  % EXPECT: A = matrix[n x (k+m)]
  ```

| Test | Description | Warns |
|------|-------------|-------|
| **Basics** (7) | | |
| basics/valid_add.m | Valid matrix addition | 0 |
| basics/invalid_add.m | Invalid addition (dimension mismatch) | 1 |
| basics/matrix_multiply.m | Valid matrix multiplication | 0 |
| basics/inner_dim_mismatch.m | Inner dimension mismatch | 1 |
| basics/scalar_matrix_ops.m | Scalar-matrix operations | 0 |
| basics/elementwise_ops.m | Elementwise operation mismatch | 1 |
| basics/reassignment.m | Incompatible reassignment | 1 |
| **Symbolic** (2) | | |
| symbolic/dimension_tracking.m | Colon vectors and symbolic dimensions | 0 |
| symbolic/dimension_arithmetic.m | Symbolic dimension arithmetic | 0 |
| **Indexing** (7) | | |
| indexing/scalar_index.m | Scalar indexing | 0 |
| indexing/slice_index.m | Slice indexing | 0 |
| indexing/range_index.m | Range indexing | 0 |
| indexing/linear_index.m | Linear indexing (scalar base error) | 1 |
| indexing/invalid_row_index.m | Constant range row indexing | 0 |
| indexing/invalid_col_index.m | Constant range column indexing | 0 |
| indexing/invalid_linear_index.m | Non-scalar index argument | 1 |
| **Control Flow** (15) | | |
| control_flow/if_branch_mismatch.m | Control-flow joins with mismatches | 1 |
| control_flow/if_else_error_branch.m | If-else with error in branch | 1 |
| control_flow/suspicious_comparison.m | Matrix-scalar comparisons | 1 |
| control_flow/elseif_chain.m | If-elseif-else chain | 0 |
| control_flow/elseif_no_else.m | Elseif without else | 0 |
| control_flow/elseif_mismatch.m | Elseif branch mismatches | 0 |
| control_flow/break_simple.m | Break in for loop | 0 |
| control_flow/continue_simple.m | Continue in while loop | 0 |
| control_flow/break_nested_loop.m | Break in nested loops | 0 |
| control_flow/switch_basic.m | Switch/case with otherwise | 0 |
| control_flow/switch_no_otherwise.m | Switch without otherwise | 0 |
| control_flow/switch_mismatch.m | Switch case shape conflicts | 0 |
| control_flow/try_catch_basic.m | Try/catch exception handling | 1 |
| control_flow/try_catch_no_error.m | Try/catch with no error | 0 |
| control_flow/try_nested.m | Nested try/catch | 0 |
| **Literals** (7) | | |
| literals/matrix_literal.m | Basic matrix literals | 0 |
| literals/horzcat_vertcat.m | Horizontal and vertical concatenation | 1 |
| literals/symbolic_concat.m | Symbolic concatenation | 0 |
| literals/string_literal.m | String literals (both quote styles) | 0 |
| literals/string_horzcat.m | String concatenation via horzcat | 0 |
| literals/string_matrix_error.m | String-matrix arithmetic warning | 1 |
| literals/string_in_control_flow.m | String/scalar join across branches | 0 |
| **Builtins** (8) | | |
| builtins/unknown_function.m | Unknown function warning | 1 |
| builtins/shape_preserving.m | size() and isscalar() builtins | 0 |
| builtins/call_vs_index.m | Apply node disambiguation | 1 |
| builtins/apply_disambiguation.m | Environment-based disambiguation | 0 |
| builtins/constructors.m | Matrix constructors and element-wise builtins | 0 |
| builtins/dim_arithmetic.m | Dimension arithmetic in builtin args | 0 |
| builtins/reshape_repmat.m | reshape and repmat shape rules | 0 |
| builtins/remaining_builtins.m | det/diag/inv/linspace/norm | 0 |
| **Loops** (22) | | |
| loops/for_loop_var.m | Loop variable bound to scalar | 0 |
| loops/for_range_symbolic.m | For loop with symbolic range | 0 |
| loops/nested_loop.m | Nested for loops | 0 |
| loops/while_basic.m | While loop with condition | 0 |
| loops/simple_accumulation.m | Single-pass misses feedback | 1 |
| loops/matrix_growth.m | Matrix growth with symbolic iteration | 1 |
| loops/loop_exit_join.m | Variables outside loop unaffected | 0 |
| loops/loop_may_not_execute.m | Post-loop join preserves pre-loop state | 1 |
| loops/warning_dedup.m | Warning deduplication in loops | 1 |
| loops/fixpoint_convergence.m | Fixed-point convergence | 1 |
| loops/widen_col_grows.m | Column dimension grows, row stable | 1 |
| loops/widen_multiple_vars.m | Multiple variables with mixed stability | 1 |
| loops/widen_self_reference.m | Self-referencing doubling | 1 |
| loops/widen_while_growth.m | While loop with matrix growth | 1 |
| loops/widen_concat_unknown.m | Concat with unknown in loop | 1 |
| loops/widen_error_in_branch.m | Unknown function in loop branch | 2 |
| loops/widen_first_assign_in_body.m | Variable first assigned in loop | 0 |
| loops/widen_if_in_loop.m | Conditional growth in loop | 1 |
| loops/widen_interdependent_vars.m | Interdependent variables in loop | 2 |
| loops/widen_stable_overwrite.m | Stable overwrite in loop | 1 |
| loops/widen_unknown_false_positive.m | Unknown function false positive | 1 |
| loops/widen_unknown_in_body.m | Unknown function overwrites in loop | 1 |
| **Functions** (39) | | |
| functions/simple_function.m | Basic single-return function | 0 |
| functions/multiple_returns.m | Multi-return with destructuring | 0 |
| functions/matrix_constructor.m | Function with dimension arguments | 0 |
| functions/procedure.m | Procedure (no return values) | 1 |
| functions/unknown_in_function.m | Unknown shapes propagate to caller | 1 |
| functions/function_then_script.m | Function definitions then script | 0 |
| functions/call_with_mismatch.m | Call with incompatible arg shape | 1 |
| functions/recursion.m | Recursive function (guard) | 1 |
| functions/cache_hit.m | Polymorphic cache hit | 0 |
| functions/cache_miss.m | Polymorphic cache miss | 1 |
| functions/cache_hit_with_warning.m | Warning replay on cache hit | 2 |
| functions/cache_symbolic_args.m | Cache with symbolic dimensions | 0 |
| functions/cache_warning_replay.m | Warning replay with line numbers | 2 |
| functions/return_statement.m | Return statement (early exit) | 0 |
| functions/return_in_script.m | Return in script context | 1 |
| functions/return_in_if.m | Return inside if-branch | 0 |
| functions/return_in_loop.m | Return inside loop body | 0 |
| functions/early_return_multi_output.m | Early return in multi-output function | 0 |
| functions/function_in_loop.m | Function call inside loop | 0 |
| functions/nested_function_calls.m | Nested function calls | 0 |
| functions/procedure_with_return.m | Procedure with explicit return | 1 |
| functions/arg_count_mismatch_cached.m | Arg count mismatch | 1 |
| functions/lambda_basic.m | Anonymous function definition | 0 |
| functions/lambda_zero_args.m | Zero-argument lambda definition | 0 |
| functions/lambda_call_approximate.m | Lambda call with body analysis | 0 |
| functions/function_handle_from_name.m | Named function handle (`@myFunc`) | 0 |
| functions/function_handle_join.m | Function handle join in control flow | 0 |
| functions/lambda_store_retrieve.m | Lambda storage with distinct IDs | 0 |
| functions/lambda_closure_capture.m | Closure captures environment by-value | 0 |
| functions/lambda_call_basic.m | Lambda body analysis with shape inference | 0 |
| functions/lambda_call_closure.m | Lambda uses closure variable | 0 |
| functions/lambda_polymorphic_cache.m | Same lambda, different arg shapes | 0 |
| functions/lambda_recursive.m | Self-referencing lambda (recursion guard) | 1 |
| functions/lambda_arg_count_mismatch.m | Lambda argument count mismatch | 1 |
| functions/lambda_dim_aliasing.m | Dimension aliasing through lambda | 0 |
| functions/lambda_zero_args_call.m | Zero-argument lambda call | 0 |
| functions/lambda_control_flow_join.m | Lambda in if/else, joined call | 0 |
| functions/handle_dispatch_builtin.m | Handle dispatches to builtin | 0 |
| functions/handle_dispatch_user_func.m | Handle dispatches to user function | 0 |
| **Structs** (5) | | |
| structs/struct_create_assign.m | Struct creation via field assignment | 0 |
| structs/struct_field_access.m | Struct field access | 0 |
| structs/struct_field_not_found.m | Missing field warning | 1 |
| structs/struct_field_reassign.m | Field reassignment with different shape | 0 |
| structs/struct_in_control_flow.m | Struct join across branches | 0 |
| **Cells** (9) | | |
| cells/cell_literal.m | Cell array literal syntax | 0 |
| cells/cell_indexing.m | Curly-brace cell indexing | 0 |
| cells/cell_assignment.m | Cell element assignment | 0 |
| cells/cell_builtin.m | cell() constructor builtin | 0 |
| cells/cell_in_control_flow.m | Cell array join across branches | 0 |
| cells/cell_mixed_types.m | Cells with mixed element types | 0 |
| cells/cell_symbolic_dims.m | Cells with symbolic dimensions | 0 |
| cells/cell_transpose.m | Cell array transpose | 0 |
| cells/curly_indexing_non_cell.m | Curly indexing on non-cell (error) | 1 |
| **Recovery** (6) | | |
| recovery/struct_field.m | Field access on non-struct | 1 |
| recovery/cell_array.m | Curly indexing on non-cell value | 1 |
| recovery/multiple_assignment.m | Unsupported multiple assignment | 1 |
| recovery/multiline_braces.m | Unsupported multiline cell indexing | 1 |
| recovery/dot_elementwise.m | Dot-elementwise edge cases | 0 |
| recovery/end_in_parens.m | End inside parentheses | 1 |

## Getting Started

Clone and Verify:
```bash
git clone <repo-url>
cd matlab-static-dimension-analysis
make test
```

Analyze a File:
```bash
make run FILE=tests/basics/inner_dim_mismatch.m
```

Example Output:
```
Warnings:
  - Line 11: Dimension mismatch in expression (A * x):
    inner dims 4 vs 5 (shapes matrix[3 x 4] and matrix[5 x 1])

Final environment:
    Env{A: matrix[3 x 4], x: matrix[5 x 1], y: unknown}
```

## CLI Options

`mmshape.py file.m` – analyze a file (IR-based)

`--tests` – run full test suite

`--strict` – fail on unsupported constructs

`--fixpoint` – use fixed-point iteration for loop analysis

Exit codes:

`0` – success

`1` – parse error, analyzer mismatch, or test failure

## Notes and Challenges

- The IR enforces structural invariants absent in the raw syntax AST

- Matrix literals are parsed with MATLAB-aware rules.

- The analyzer uses best-effort inference; Even when a definite mismatch is detected, it continues analysis to provide as much information as possible.

- The analyzer is strict on provable dimension errors. When an operation is definitely invalid (e.g., inner-dimension mismatch in A*B), it emits a warning and treats the expression result as unknown.

## Limitations
This tool does not support:
- File I/O
- Plotting or graphics
- Precise loop invariants
- Nested functions

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

Possible future extensions include:

- Nested functions
- Stricter invalidation semantics for definite errors
- Richer symbolic constraint solving
- IDE or language-server integration
