# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Ambiguified test counts in documentation (CLAUDE.md, AGENTS.md, README.md)
- quality-assurance agent for code quality and project hygiene
- documentation-maintainer agent for documentation synchronization
- release-coordinator agent for release orchestration
- structural-ci-gatekeeper agent (renamed from test-runner1)
- semantic-differential-auditor agent (renamed from test-runner2)
- Comprehensive agent workflow documentation in AGENTS.md
- Agent design analysis document

### Changed
- Renamed test-runner1 to structural-ci-gatekeeper for clarity
- Renamed test-runner2 to semantic-differential-auditor for clarity
- Enhanced validation pipeline with quality, structural, and semantic checks

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
- CLAUDE.md project documentation for AI-assisted development
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
