# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2026-02-14
### Added
- `SymDim` frozen dataclass: canonical polynomial representation for symbolic dimensions
- Multivariate polynomial algebra: addition, multiplication, negation, subtraction with automatic normalization
- Like-term collection: `n+n` canonicalizes to `2*n`
- Commutativity: `n+m` and `m+n` are now structurally equal
- Associativity: `(n+m)+k` flattens to `k+m+n` (degree-descending, alphabetical)
- Constant-offset conflict detection: `dims_definitely_conflict(n+1, n)` returns `True`
- 3 new test files in tests/symbolic/ (134 total, was 131)

### Changed
- `Dim` type from `Union[int, str, None]` to `Union[int, SymDim, None]`
- `add_dim`/`mul_dim` produce `SymDim` for symbolic cases (with automatic constant folding to `int`)
- `dims_definitely_conflict` checks if difference is a nonzero constant (more precise)
- `expr_to_dim_ir` returns `SymDim.var(name)` for symbolic variables
- Subtraction simplified: `add_dim(left, mul_dim(-1, right))` replaces string manipulation
- Display: degree-descending, alphabetical within degree, constant last (e.g., `m*n+2*n+3`)
- `dim_aliases` type widened from `Dict[str, str]` to `Dict[str, Any]`
- 4 test expectations updated for canonical display forms

## [0.12.3] - 2026-02-14
### Added
- Cell element assignment: `c{i} = expr` and `c{i,j} = expr` syntax
- `CellAssign` IR node (mirrors StructAssign pattern)
- W_CELL_ASSIGN_NON_CELL warning for curly assignment on non-cell values
- Bottom-to-cell promotion: unbound `x{i} = expr` creates `cell[None x None]`
- 4 new test files in tests/cells/ (131 total, was 127)

## [0.12.2] - 2026-02-14
### Added
- Cell array support: 8th shape kind `cell[r x c]` tracking container dimensions
- Cell literal syntax: `{1, 2; 3, 4}` parsed via shared `_parse_delimited_rows` helper
- Curly-brace content indexing: `c{i}`, `c{i,j}` returns `unknown` (element shapes deferred to v0.13.0+)
- `cell(n)` and `cell(m, n)` builtin constructors with symbolic dimension support
- `CellLit` and `CurlyApply` IR nodes (mirrors MatrixLit/Apply pattern)
- W_CURLY_INDEXING_NON_CELL warning for curly indexing on non-cell values
- `pretty_expr_ir` support for CellLit and CurlyApply in diagnostics
- 9 new test files in tests/cells/ (127 total, was 118)
- New test category: tests/cells/ (11th category)

### Changed
- Extracted `_parse_delimited_rows(end_token)` from `parse_matrix_literal` (shared by matrix and cell literals)
- Lexer: added `}` to transpose context set (`c{1}'` now correctly lexed as transpose)
- `starts_expr` updated to include `{` (cell literals in implicit column separation)
- `join_shape` and `widen_shape` handle cell kind (pointwise on dimensions)
- README refreshed: removed "Mini-MATLAB" framing, added Warns column to test table, updated to 127 tests

## [0.12.1] - 2026-02-14
### Added
- Lambda body analysis at call sites with polymorphic caching (replaces W_LAMBDA_CALL_APPROXIMATE approximation)
- Function handle dispatch: `@myFunc` resolves to user-defined functions, `@builtin` resolves to builtins
- Closure metadata storage in AnalysisContext with lambda ID tracking via frozenset
- Control flow join precision: if/else branches with different lambdas both analyzed, results joined
- Dimension aliasing through lambda calls (symbolic dimensions propagate from caller to lambda body)
- Recursion guard for self-referencing lambdas
- W_LAMBDA_ARG_COUNT_MISMATCH warning for lambda arity mismatches
- W_RECURSIVE_LAMBDA warning for recursive lambda calls
- 12 new test files in tests/functions/ (118 total, was 106)

### Changed
- Shape dataclass: added `_lambda_ids` field (Optional[frozenset]) for tracking lambda identity
- `join_shape` and `widen_shape` now union lambda_ids for function_handle shapes
- AnalysisContext: replaced `_lambda_closures` with `_lambda_metadata`, added `_handle_registry` and `analyzing_lambdas`
- Dimension aliasing in `analyze_function_call` generalized to any IndexExpr (not just Var)
- W_LAMBDA_CALL_APPROXIMATE no longer emitted for lambdas with known bodies
- README test table reorganized by category with complete 118-test inventory

## [0.12.0] - 2026-02-14
### Added
- **Phase 1: String literals** with context-sensitive lexer for `'` disambiguation (string start vs transpose)
- String shape kind: `Shape.string()` for char array literals (both `'hello'` and `"hello"` syntax)
- String concatenation via horzcat: `['hello', ' ', 'world']` returns string
- String arithmetic semantics: `'hello' + ' '` produces numeric row vector (MATLAB-faithful)
- W_STRING_ARITHMETIC warning for invalid string operations (string + matrix/scalar with non-`+` operators)
- 4 new test files in tests/literals/ (string_literal, string_horzcat, string_matrix_error, string_in_control_flow)
- **Phase 2: Struct support** with field access and assignment
- Struct shape kind: `Shape.struct(fields)` with sorted-tuple storage for hashability
- Chained dot access: `s.a.b = 5`, `x = s.a.b` (nested FieldAccess IR nodes)
- Union-with-bottom join semantics: missing fields get `Shape.bottom()` (identity in join, not `unknown`)
- FieldAccess IR node for struct field reads (chained via nested nodes)
- StructAssign IR statement for struct field writes (`s.a.b = expr`)
- W_STRUCT_FIELD_NOT_FOUND warning when accessing non-existent struct field
- W_FIELD_ACCESS_NON_STRUCT warning when using dot access on non-struct value
- 5 new test files in tests/structs/ (struct_create_assign, struct_field_access, struct_field_not_found, struct_in_control_flow, struct_field_reassign)
- New test category: tests/structs/ (10th category)
- **Phase 3: Anonymous functions and function handles**
- Function handle shape kind: `Shape.function_handle()` for lambdas and named handles
- Lambda IR node for anonymous function expressions: `@(x, y) expr`
- FuncHandle IR node for named function handles: `@myFunc`
- Closure tracking in AnalysisContext with monotonic closure counter (avoids line number collision)
- Lambda calls emit W_LAMBDA_CALL_APPROXIMATE and return unknown (body analysis deferred to v0.12.1)
- Function handle variables shadow builtins in Apply disambiguation (matches MATLAB scoping)
- 5 new test files in tests/functions/ (lambda_basic, lambda_call_approximate, lambda_zero_args, function_handle_from_name, function_handle_join)
- Total test count: 106 (was 92)

### Changed
- Promoted tests/recovery/struct_field.m from W_UNSUPPORTED_STMT to W_FIELD_ACCESS_NON_STRUCT (recovery→first-class support)
- Apply node disambiguation: function_handle variables override KNOWN_BUILTINS lookup (handle shadows builtin)
- Lexer: context-sensitive `'` token handling based on previous token (ID/`)'/`]`/NUMBER → transpose, else → string start)
- Test categories: 10 directories (added structs/)

## [0.11.0] - 2026-02-14
### Added
- Extended control flow constructs: elseif, break, continue, switch/case, try/catch
- 8 new parser keywords: elseif, break, continue, switch, case, otherwise, try, catch
- 5 new IR nodes: IfChain (replaces If), Switch, Try, Break, Continue
- Exception handling: EarlyBreak and EarlyContinue exceptions for loop control flow
- Full exception handling in all branch handlers (IfChain, Switch, Try)
- 12 new test files in tests/control_flow/ (elseif chains, break/continue, switch/case, try/catch)
- Total test count: 92 (was 80)

### Changed
- If statement now supports elseif chains via IfChain IR node
- Loop analysis catches EarlyBreak and EarlyContinue exceptions

## [0.10.2] - 2026-02-14
### Added
- Test coverage polish: 11 new test files for edge cases and integration scenarios
- Return statement edge cases: return_in_if.m (return in if-branch), return_in_loop.m (return in loop body), early_return_multi_output.m (partial output assignment)
- Cache edge cases: cache_symbolic_args.m (symbolic dimension cache keys), cache_warning_replay.m (warning replay verification), arg_count_mismatch_cached.m (arg count error without cache interaction)
- Integration tests: function_in_loop.m (cache in loop), nested_function_calls.m (non-recursive nesting), procedure_with_return.m (procedure early exit)
- Extended existing tests: return_statement.m (added unreachable code case), multiple_returns.m (added single-output usage of multi-output function)
- Total test count: 80 (was 71)

## [0.10.1] - 2026-02-14
### Added
- Polymorphic caching for function analysis keyed by (func_name, arg_shapes)
- Warning replay: Cached warnings replayed with current call-site line number
- Return statement support (keyword, IR node, EarlyReturn exception)
- Catch-at-boundary semantics: EarlyReturn caught in If, loops, and program boundaries
- W_RETURN_OUTSIDE_FUNCTION warning for return in script context
- 5 new test files in tests/functions/ (cache_hit, cache_miss, cache_hit_with_warning, return_statement, return_in_script)
- Total test count: 71 (was 66)

### Changed
- AnalysisContext dataclass: Added analysis_cache field (Dict[tuple, tuple])
- Function analysis: Re-analysis only on cache miss (arg shapes change)
- Loop analysis: Catches EarlyReturn at boundary (stops iteration, doesn't propagate)
- If statement: Catches EarlyReturn from non-returned branch
- Program analysis: Catches EarlyReturn at script level (stops analysis)

### Added (Infrastructure)
- _format_dual_location_warning() helper: Extracted for reuse in warning replay
- Return IR node: Represents return statement (no value in MATLAB)
- EarlyReturn exception: Signals early function exit
- KEYWORDS updated: Added "return" keyword to parser

## [0.10.0] - 2026-02-14
### Added
- User-defined function support with 3 syntactic forms (single return, multi-return, procedure)
- Interprocedural shape inference (call-site specific analysis with caller's argument shapes)
- Dimension aliasing across function boundaries (symbolic dimension names propagate)
- Destructuring assignment for multiple return values (`[a, b] = func(x)`)
- AnalysisContext dataclass for threading analysis state (function registry, recursion guard, fixpoint flag)
- FunctionSignature dataclass for registered function metadata
- Two-pass program analysis (Pass 1: register functions, Pass 2: analyze script)
- Recursion guard to prevent infinite loops on recursive calls
- Dual-location warnings for function bodies (shows both call site and body location)
- 6 new warning codes: W_FUNCTION_ARG_COUNT_MISMATCH, W_RECURSIVE_FUNCTION, W_PROCEDURE_IN_EXPR, W_MULTI_ASSIGN_NON_CALL, W_MULTI_ASSIGN_BUILTIN, W_MULTI_ASSIGN_COUNT_MISMATCH
- 8 new test files in `tests/functions/` category
- Total test count: 66 (was 58)

### Changed
- IR nodes: Added FunctionDef and AssignMulti
- Test categories: Now 9 directories (added `functions/` to existing 8)
- Apply node disambiguation now checks user-defined function registry

## [0.9.3] - 2026-02-14
### Fixed
- Soundness bug: `Shape.unknown()` conflated unbound variables (bottom) with errors (top)
- `widen_shape(matrix, error_unknown)` now correctly returns unknown (was returning matrix)
- False positive: unknown function in loop no longer causes spurious downstream warnings in fixpoint mode

### Changed
- `Env.get()` returns `Shape.bottom()` for unbound variables (was `Shape.unknown()`)
- `join_shape(unknown, X)` now returns unknown for soundness (precision regression, more sound)
- `widen_shape`/`join_shape` lattice semantics: bottom is identity, unknown is absorbing top

### Added
- `Shape.bottom()` as distinct lattice element (identity in join/widen)
- 6 new edge case tests for bottom/error distinction
- Total test count: 58 (was 52)

## [0.9.2] - 2026-02-13
### Added
- Principled loop widening with guaranteed convergence (≤2 iterations)
- Widening operators: `widen_dim()`, `widen_shape()`, and `widen_env()` in runtime/
- 3-phase widening algorithm in `_analyze_loop_body()`: discover, stabilize, post-loop join
- Per-dimension widening: stable dimensions preserved, conflicting dimensions → None
- 4 new loop tests for widening behavior (widen_col_grows, widen_multiple_vars, widen_self_reference, widen_while_growth)

### Changed
- Replaced iteration-limit loop analysis (MAX_LOOP_ITERATIONS=3) with widening-based fixpoint
- `widen_shape()` treats unknown as bottom (superseded by Shape.bottom() in 0.9.3)
- Same `widen_env()` operator used for both widening step and post-loop join
- Updated 3 existing loop tests (EXPECT_FIXPOINT warnings: 3→1 for growth patterns)

## [0.9.1] - 2026-02-13
### Added
- Dimension arithmetic in builtin arguments: `expr_to_dim_ir` now evaluates `+`, `-`, `*` in dimension expressions
- Support for `zeros(n+1, m)`, `reshape(A, 2*n, m+1)`, etc. with symbolic arithmetic
- Concrete arithmetic folding (e.g., `zeros(2+3, 4*2)` → `matrix[5 x 8]`)
- Test file `tests/builtins/dim_arithmetic.m` with 9 dimension arithmetic cases
- Shape rules for 5 new builtins: `det` (→scalar), `diag` (vector↔diagonal matrix), `inv` (pass-through for square), `linspace` (→row vector), `norm` (→scalar)
- Complete builtin shape rule coverage: `BUILTINS_WITH_SHAPE_RULES == KNOWN_BUILTINS` (19/19)
- Test file `tests/builtins/remaining_builtins.m` with 19 expectations for new builtins

### Fixed
- `zeros(n)` and `ones(n)` 1-arg forms now correctly return `matrix[n x n]` instead of unknown

### Removed
- `--compare` mode and `make compare` target (legacy analyzer no longer tested)
- `Call` and `Index` IR nodes (unified under `Apply`)

### Changed
- Moved `KNOWN_BUILTINS` from parser to `analysis/builtins.py` (fixed layering violation)

### Added (Infrastructure)
- `analysis/builtins.py`: centralized builtin function catalog
- Test `tests/builtins/apply_disambiguation.m` for Apply node edge cases

## [0.8.6] - 2026-02-13
### Added
- Fixed-point iteration for loop analysis via `--fixpoint` CLI flag (max 3 iterations with convergence detection)
- `EXPECT_FIXPOINT:` test directive for fixpoint-specific expectations
- New test category `tests/loops/` with 10 test files

### Fixed
- `For` loop handler now binds loop variable to scalar shape
- Warning deduplication preserves first-occurrence order via `dict.fromkeys`

## [0.8.5] - 2026-02-13
### Added
- Shape rules for `reshape(A, m, n)` → `matrix[m x n]` and `repmat(A, m, n)` → `matrix[r*m x c*n]`
- `mul_dim()` function in runtime/shapes.py for multiplicative symbolic arithmetic (with 0/1 short-circuits)
- Test file tests/builtins/reshape_repmat.m with 7 test cases

## [0.8.4] - 2026-02-13
### Added
- Rich builtin shape rules for 12 functions (matrix constructors, element-wise, query, transpose)
- Matrix constructors: `eye`, `rand`, `randn` (0-arg→scalar, 1-arg→n×n, 2-arg→m×n)
- Element-wise functions: `abs`, `sqrt` (pass-through shape)
- Transpose function: `transpose()` (swaps row/col, consistent with `.'` operator)
- Query functions: `length`, `numel` (return scalar)
- `_eval_index_arg_to_shape()` helper for evaluating IndexArg to Shape in analysis/analysis_ir.py
- Test31.m with 38 assertions covering all new builtin shape rules

### Fixed
- Non-deterministic output in `join_env()` (runtime/env.py now uses sorted iteration)

### Changed
- Updated test28.m expectation for `randn(3,4)` → `matrix[3 x 4]`

## [0.8.3] - 2026-02-13
### Changed
- Extracted shared indexing logic into `_eval_indexing()` helper in analysis/analysis_ir.py
- Consolidated three duplicate copies of indexing logic (Apply-colon, Apply-default, Index handler)
- Typed Apply.args properly as `List[IndexArg]` in ir/ir.py (was bare `List`)

## [0.8.2] - 2026-02-13
### Added
- Unified `Apply` IR node for call-vs-index disambiguation at analysis time
- Parser now emits `['apply', ...]` for all `foo(...)` expressions (no parse-time decision)
- `unwrap_arg()` helper in analysis/analysis_ir.py for extracting Expr from IndexArg
- Test30.m for Apply node disambiguation (colon→indexing, unbound→W_UNKNOWN_FUNCTION, bound→indexing)

### Changed
- Call-vs-index disambiguation moved from parse time (parser) to analysis time (analyzer)
- Analyzer uses colon/range presence, builtin whitelist, and variable binding to disambiguate
- `pretty_expr_ir` in analysis/diagnostics.py now handles Apply nodes

## [0.8.1] - 2026-02-13
### Added
- Expanded builtin function set from 4 to 19 functions (KNOWN_BUILTINS in frontend/matlab_parser.py)
- New warning code W_UNKNOWN_FUNCTION for unrecognized function calls (informational)
- Test28.m for expanded builtins and unknown function warning

### Changed
- Unrecognized function calls now return unknown shape instead of scalar (soundness fix)
- Previous test28.m renumbered to test29.m

## [0.7e] - 2026-02-05
### Added
- Support for multiline matrix literals with newline row separators
- Enhanced matrix literal parsing in frontend/matlab_parser.py

### Fixed
- Matrix literal shape inference for multi-row matrices

## [0.7d] - 2026-01-28
### Added
- Stable warning codes with W_* prefix convention
- Strict-mode policy moved to CLI for better control
- Warning code documentation in diagnostics.py

### Changed
- CLI now handles strict mode validation centrally

## [0.7c] - 2026-01-15
### Added
- Parse recovery for unsupported constructs
- OpaqueStmt IR node for unhandled statements
- Glob-based test discovery (dynamic test file detection)
- Tests for unsupported construct recovery (test22.m-test27.m)

### Changed
- Test suite now uses glob pattern matching
- Parser continues after encountering unsupported constructs

## [0.6] - 2025-12-01
### Added
- Comprehensive docstrings for all public functions
- Type hints throughout codebase
- Detailed function documentation in all modules

### Changed
- Improved naming consistency across codebase
- Enhanced code organization

### Removed
- Dead code cleanup across all modules
- Unused imports and variables

## [0.5] - 2025-11-15
### Added
- IR AST pipeline (dataclass-based)
- Consolidated CLI interface (mmshape.py)
- Frontend lowering from syntax AST to IR AST

### Changed
- IR-based analyzer is now the default
- Legacy analyzer moved to legacy/ directory

## [0.4] - 2025-11-01
### Added
- Symbolic dimension tracking (n, m, k variables)
- Symbolic dimension arithmetic for concatenation
- Control flow joins (if/else, while)

### Changed
- Shape inference enhanced with symbolic dimensions

## [0.3] - 2025-10-15
### Added
- Matrix literal parsing
- Horizontal and vertical concatenation
- Colon vector support (1:n)

## [0.2] - 2025-10-01
### Added
- Matrix multiplication shape checking
- Transpose operator
- Indexing support (A(i,j), slicing)

## [0.1] - 2025-09-15
### Added
- Initial parser for Mini-MATLAB subset
- Basic shape inference for scalar and matrix types
- Simple arithmetic operations (+, -, *)
- Test suite infrastructure
