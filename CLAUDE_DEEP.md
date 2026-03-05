# CLAUDE_DEEP.md

This file contains the deeper, long-form repository documentation. It is not meant to be pasted into every agent run. Pull it in when you are doing architecture-level work, deep debugging, or you need the exact details of semantics.

## Overview

This project implements a static shape and dimension analysis for MATLAB. The analyzer detects matrix-related errors (dimension mismatches, incompatible operations) before runtime by reasoning about matrix shapes, symbolic dimensions, and control flow without executing code.

**Key principle**: The F# analyzer (`src/`) is the sole implementation. The Python codebase was deleted on 2026-02-22.

## Essential Commands

**Run all tests (F#)**:
```bash
cd src && dotnet run -- --tests
```

**Analyze a single file**:
```bash
cd src && dotnet run -- ../tests/basics/inner_dim_mismatch.m
```

**Strict / fixpoint / witness / pro modes**:
```bash
cd src && dotnet run -- --strict ../tests/recovery/struct_field.m
cd src && dotnet run -- --fixpoint ../tests/loops/matrix_growth.m
cd src && dotnet run -- --witness ../tests/basics/inner_dim_mismatch.m
cd src && dotnet run -- --license CONF-xxx.yyy ../tests/intervals/index_out_of_bounds.m
```

**Build VS Code extension (Fable)**:
```bash
cd src && dotnet fable ../vscode-conformal/fable/ConformalFable.fsproj
cd vscode-conformal && node esbuild.mjs
cd vscode-conformal && npx @vscode/vsce package --allow-missing-repository
```

**Run F# LSP server (native .NET)**:
```bash
cd src && dotnet run -- --lsp
```

## Architecture

The project has an F# analyzer and a Fable compilation layer for the VS Code extension:

### F# Analyzer (`src/`) — 38 files, ~13.2K lines (Sprint I+J)

Key files:
- `Ir.fs`: Discriminated unions for Expr and Stmt (the IR)
- `Lexer.fs`: Tokenizer (mutable position, regex-based; `#if FABLE_COMPILER` guards for JS compat)
- `Parser.fs`: Recursive-descent parser, emits typed IR nodes directly
- `Shapes.fs`: Shape domain (DU-based: Scalar, Matrix, String, Struct, FunctionHandle, Cell, Unknown, Bottom)
- `SymDim.fs`: Symbolic dimension polynomials with rational coefficients
- `Env.fs`: Variable environment with parent-pointer scope chains
- `EvalExpr.fs`: Expression evaluation and indexing logic
- `EvalBinop.fs`: Binary operation shape rules
- `EvalBuiltins.fs`: Builtin function evaluation
- `StmtFuncAnalysis.fs`: Statement dispatch and function/lambda analysis
- `Analysis.fs`: Entry point, two-pass analysis
- `SharedTypes.fs`: Type-safe AnalysisContext definition (no obj boxing)
- `Context.fs`: AnalysisContext with mutable dictionaries
- `Workspace.fs`: Cross-file workspace scanning and analysis (`#if FABLE_COMPILER` guards for System.IO)
- `Diagnostics.fs`: Warning codes, strict-only classification
- `Intervals.fs`: Integer interval domain with symbolic bounds and condition refinement
- `DimEquiv.fs`: Dimension equivalence classes (union-find) for backward constraint propagation
- `Constraints.fs`: Dimension constraint tracking (uses immutable F# `Map`/`Set` on `ConstraintContext`; `SnapshotScope` is O(1))
- `DimExtract.fs`: Dimension extraction from expressions
- `EndHelpers.fs`: End keyword resolution utilities
- `PathConstraints.fs`: Branch-aware path constraint stack
- `MatrixLiterals.fs`: Matrix literal shape inference
- `Witness.fs`: Incorrectness witness generation
- `Builtins.fs`: Builtin function catalog (KNOWN_BUILTINS)
- `AnalysisCore.fs`: Shared compatibility checks
- `Json.fs`: JSON serialization utilities
- `PropertyTests.fs`: FsCheck property-based tests (28 properties, 6 sections)
- `LspServer.fs`: Native .NET LSP server (Ionide.LanguageServerProtocol)
- `LspDiagnostics.fs`: LSP diagnostic conversion with severity mapping
- `LspHover.fs`: Hover provider showing inferred shapes at cursor
- `LspSymbols.fs`: Document symbol provider for breadcrumbs/outline
- `LspDefinition.fs`: Go-to-definition provider
- `LspCodeActions.fs`: Quick-fix code actions (`*` -> `.*`, `&&` -> `&`, `||` -> `|`)
- `Cli.fs`: Command-line interface
- `TestRunner.fs`: Test infrastructure
- `Program.fs`: Entry point

Build: `cd src && dotnet build` / Test: `cd src && dotnet run -- --tests`

### Fable Compilation Layer (`vscode-conformal/fable/`)

Fable compiles 27 core F# files to JavaScript for the VS Code extension:
- `ConformalFable.fsproj`: Project file referencing 26 core .fs files from `src/` (excludes LSP/CLI/Test/PropertyTests) plus `Interop.fs`
- `Interop.fs`: TypeScript-callable API (`analyzeSource` function)
- Output: `vscode-conformal/src/fable-out/` (gitignored)
- Cross-file body analysis: external files are pre-parsed by `tryParseExternalBody` in `Interop.fs`, bodies stored on `ExternalSignature`, used by `loadExternalFunction` in the Fable path

### VS Code Extension (`vscode-conformal/`) — v2.8.0, Fable-powered

- Runs F#-compiled-to-JavaScript analyzer in-process via Node.js module transport (no external dependencies)
- `src/extension.ts`: LanguageClient with TransportKind.ipc, status bar, commands
- `src/server.ts`: LSP server using `vscode-languageserver` + Fable-compiled analyzer (`require('./fable-out/Interop.js')`)
- `esbuild.mjs`: Bundles client + server into `out/extension.js` and `out/server.js`
- Configuration settings: fixpoint, strict, pro, analyzeOnChange, inlayHints
- Commands: analyzeFile, toggleFixpoint, toggleStrict, restartServer
- 190KB VSIX; registers `.m` language with TextMate grammar (expanded: classdef, persistent, global, parfor, spmd keywords, 80+ builtins)
- Auto-reclassifies `.m` files from Objective-C to MATLAB via `firstLine` + `reclassifyIfMatlab()`

### Diagnostic System

All warnings are structured as `Diagnostic` record instances:
- **Fields**: `line`, `col`, `code` (W_* prefix), `message`, optional `related_line`, `related_col`
- **Warning codes**: All warnings have stable W_* codes
- **Warning tiers**: `STRICT_ONLY_CODES` in `Diagnostics.fs` defines 11 codes suppressed in default mode (shown only with `--strict`); `PRO_ONLY_CODES` defines 11 codes that require the advanced analysis domains and are suppressed unless a valid license key is provided via `--license` (the CLI prints an upsell count when pro-tier warnings exist but no license is active); the LSP reads `conformal.licenseKey` and validates it via Ed25519 signature to determine pro access. Filtering is a post-analysis presentation concern applied in CLI and LSP. Tests receive unfiltered warnings.
- **Severity mapping** (LSP): Error-severity codes defined in `LspDiagnostics.fs` and `server.ts`. W_UNSUPPORTED_* codes get DiagnosticTag.Unnecessary.

## Shape System

Each expression gets a shape from this abstract domain:
- **scalar**: Single values (e.g., `5`, scalar variables)
- **matrix[r x c]**: Where `r` and `c` can be:
  - Concrete integers (e.g., `3`, `4`)
  - Symbolic expressions (e.g., `n`, `m`, `k`, `n+m`, `2*n`)
  - Range dimensions (`Range(lo, hi)`) for interval-valued dimensions
  - Unknown (`None`)
- **string**: Char array literals (`'hello'`, `"world"`)
- **struct{fields}**: Struct values with named fields (e.g., `struct{x: scalar, y: matrix[3 x 1]}`); `StructShape` has an `_open: bool` flag; open structs (created when you assign a field to an unknown base) display as `struct{x: scalar, ...}` and don't warn on missing field access; lattice ordering: bottom < closed struct < open struct < unknown
- **function_handle**: Anonymous functions (`@(x) expr`) or named handles (`@myFunc`)
  - Lambda bodies are analyzed at call sites with caller's argument shapes
  - Closure capture: lambdas capture environment by-value at definition
  - Polymorphic: same lambda called with different arg shapes is analyzed per arg tuple
  - Function handle dispatch: `@myFunc` and `@builtin` dispatch to their targets
- **cell[r x c]**: Cell arrays with optional per-element shape tracking
  - `_elements` field tracks shapes of individual elements (linear-indexed)
  - Literal indexing `c{i}` with concrete index extracts precise element shape
  - Dynamic indexing joins all element shapes conservatively
  - Element tracking dropped when `_elements = None` (absorbing in joins)
- **unknown**: Error or indeterminate shape (lattice top)
- **bottom**: Unbound variable / no information (lattice identity, internal-only -- converted to unknown at expression eval boundary)

**Key features**:
- Symbolic dimension tracking (e.g., `n`, `m` represent dimensions)
- Symbolic arithmetic for concatenation (e.g., `n x (k+m)`), multiplication (e.g., `(n*k)`), and dimension expressions (e.g., `zeros(n+1, m)`)
- Control flow joins (merges `if`/`else` branches conservatively)
- Single-pass loop analysis by default (optional widening-based fixpoint via `--fixpoint`)
- **Widening-based loop analysis**: 3-phase algorithm (discover, stabilize, post-loop join) guarantees convergence in <=2 iterations by widening conflicting dimensions to None while preserving stable dimensions
- **`end` keyword**: Resolves to last index in array/cell indexing context with arithmetic support (`c{end}`, `c{end-1}`, `A(1:end, end-2:end)`), emits `W_END_OUTSIDE_INDEXING` when used outside indexing; `end` in column position resolves to column dimension (not row)
- **Range indexing precision**: `index_arg_to_extent_ir()` computes symbolic extents for range args via `(b - a) + 1`; e.g., `A(1:k,:)` -> `matrix[k x c]`, `A(1:end-1,:)` -> `matrix[(n-1) x m]`
- **Dimension constraint solving**: Records equality constraints between symbolic dimensions during matmul, elementwise ops, and concat; validates concrete bindings against constraints; emits `W_CONSTRAINT_CONFLICT`; path-sensitive joins preserve only constraints added in all branches; function/lambda scopes are isolated
- **Dimension equivalence classes** (union-find): Backward constraint propagation via `DimEquiv.fs`; when analysis discovers `n == m`, they are merged into an equivalence class; when one resolves to a concrete value, the other does too
- **For-loop accumulation refinement** (`--fixpoint` only): Detects `A = [A; delta]` (vertcat) and `A = [A, delta]` (horzcat) patterns in loop bodies; extracts iteration count from range (`(b-a)+1`); replaces widened `None` dimension with `init_dim + iter_count * delta_dim`; conservative bailout for stepped ranges, conditional accumulation, and self-referencing deltas
- **Interval analysis**: Integer interval domain `[lo, hi]` tracks scalar value ranges in parallel with shape inference; bounds accept `SymDim` values for symbolic upper bounds (`for i = 1:n` -> `i in [1, n]`, symbolic bound); condition refinement narrows intervals inside branch bodies (`if x > 0` refines `x` to `[1, +inf]`), eliminating false-positive OOB/negative-dim warnings when guards prove safety; intervals join conservatively across branches; enables `W_DIVISION_BY_ZERO`, `W_INDEX_OUT_OF_BOUNDS`, and `W_POSSIBLY_NEGATIVE_DIM` checks; **threshold widening**: `widenInterval` snaps widened bounds to the nearest value in `{-1000, -100, -10, -1, 0, 1, 10, 100, 1000}` rather than jumping to Unbounded, keeping intervals finite after fixpoint loops; **cross-domain bridge** (`bridgeToDimEquiv` in `Intervals.fs`): when a variable's interval narrows to an exact value `[k, k]`, the bridge propagates `k` into any DimEquiv equivalence class that variable belongs to, and back into `valueRanges` for all equivalent variables; `n = size(A, 1)` where `A` has a concrete row dimension directly sets `valueRanges[n] = [dim, dim]` and calls the bridge; Phase 2 re-analysis in `analyzeLoopBody` fires when `valueRanges` changes (not only when shapes change), so scalar counters can widen past the first threshold during the fixpoint pass; **narrowing after widening** (`narrowInterval`, `narrowValueRanges` in `Intervals.fs`): Phase 2.5 re-runs the loop body once after fixpoint stabilization, then intersects the resulting intervals with the widened bounds; the pass uses a throwaway warnings buffer and restores the environment so only `valueRanges` is updated, keeping the result sound; **scope-limited widening** (`collectModifiedVars`, `restoreStableRanges` in `StmtFuncAnalysis.fs`): `collectModifiedVars` scans the loop body syntactically to identify every assigned variable; `restoreStableRanges` reverts `valueRanges` for all other variables to their pre-loop values after widening and narrowing, preventing bridge propagation from corrupting stable variable intervals; **Pentagon domain** (`applyPentagonBridge`, `applyPentagonLowerBridge` in `Intervals.fs`; `upperBounds: Map<string, string * int>` and `lowerBounds: Map<string, string * int>` on `ConstraintContext`): tracks relational upper-bound constraints `x <= y + c` and lower-bound constraints `x >= y + c`; the for-loop handler records `i <= n` when the range endpoint is a named variable, and `i >= start` when the start is a named variable; `killLowerBoundsFor` clears stale lower-bound entries on assignment; `joinLowerBounds` intersects lower-bound maps at control-flow joins; `applyPentagonBridge` fires when the upper-bound variable has an exact interval, tightening the constrained variable's upper bound; `applyPentagonLowerBridge` does the symmetric thing for lower bounds; both bridges are called at the start of each pass in `analyzeLoopBody` so they stay effective across all fixpoint phases; **while-loop Pentagon extraction** (`extractPentagonBoundsFromCondition` in `Intervals.fs`): parses while-loop conditions for relational bounds, handling `<=`, `<`, `>=`, `>` comparisons and `&&` conjunctions; the while-loop handler saves and restores both `upperBounds` and `lowerBounds` alongside `valueRanges`; **Pentagon index-bounds suppression** (`pentagonProvesInBounds`, `pentagonProvesLowerBound` in `Intervals.fs`): both called from `EvalExpr.fs` before emitting `W_INDEX_OUT_OF_BOUNDS`; `pentagonProvesInBounds` checks the concrete case (index interval upper bound matches dimension exactly) and the symbolic case (Pentagon bound variable and dimension name share a DimEquiv root); `pentagonProvesLowerBound` does the symmetric check for lower bounds, suppressing "index may be < 1" warnings when the Pentagon can prove the index is at least 1
- **Range-valued dimensions**: `Range(lo, hi)` dimension variant for interval-valued shape dimensions, used in conditional accumulation where the final dimension depends on which branches execute
- **Comparison broadcast shapes** (`EvalBinop.fs`): `==`, `~=`, `<`, `<=`, `>`, `>=` return the broadcast shape of their operands; `matrix[m x n] > 0` gives `matrix[m x n]`; scalar vs scalar gives scalar; this is required for logical indexing detection to work correctly
- **Logical indexing detection** (`EvalExpr.fs`): when the index argument to an `Apply` node is a matrix-typed expression (detected by checking the index arg's shape is a matrix, not Unknown), the result is `matrix[None x 1]` (column vector) for matrix inputs or `matrix[1 x None]` for row vector inputs; guards against Unknown==Unknown false positives

## Test File Format

Tests use inline assertions in MATLAB comments:
```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]   (override when --fixpoint active)
```

The test runner (`TestRunner.fs`) validates these expectations against analysis results. Test files are organized in `tests/` subdirectories by category (21 categories, 439 tests) and discovered dynamically via glob. Run `cd src && dotnet run -- --tests` to see the current count.

Additional directive forms:
- `% EXPECT: warnings >= N` (or `>`, `<`, `<=`): accepts comparison operators, not just `=`
- `stmt  % EXPECT_WARNING: W_CODE`: inline directive, checks that the given code fires on that exact line
- `stmt  % EXPECT_NO_WARNING: W_CODE`: inline directive, checks that the given code does NOT fire on that line
- `% EXPECT_FIXPOINT_WARNING: W_CODE` / `% EXPECT_FIXPOINT_NO_WARNING: W_CODE`: same, but only applies in `--fixpoint` mode; replaces the non-fixpoint inline directives when `--fixpoint` is active
- `% MODE: coder`: enables the Coder compatibility pass for that file
- `% MODE: strict`: enables strict mode for that file (same effect as `--strict` on the CLI)
- `% SKIP_TEST`: silently skips the file; intended for tests that require external resources or are temporarily disabled

`WarningCodes.fs` exports `codeMap` and `tryParseCode` so the test runner can validate that `W_CODE` strings in directives are real codes; unknown codes cause a `PARSE ERROR` at test load time.

The `--quiet` flag suppresses per-test output and only prints failures.

## Critical Implementation Details

### Single AST Format

The parser (`Parser.fs`) emits IR DU nodes directly. There is no intermediate syntax AST and no separate lowering step. Every node in the parse tree is a discriminated union case from `Ir.fs`.

All IR nodes carry `col: int` (1-based column offset). Diagnostics have `col` and `related_col` fields; the LSP server uses these for precise character-range highlighting.

### Apply Node Dispatch

- `Apply` is the sole IR node for both function calls and array indexing (`foo(...)`)
- Parser emits `Apply` for all parenthesized expressions (no parse-time disambiguation)
- Dispatch priority: colon/range in args -> indexing, function_handle variable -> lambda/handle call (shadows builtins), `KNOWN_BUILTINS` -> builtin call, user-defined function -> function call, external workspace function -> cross-file analysis (returns inferred shape), bound variable -> indexing, unknown name -> `W_UNKNOWN_FUNCTION`

### Workspace Awareness (Cross-File Analysis)

When analyzing a `.m` file, the analyzer scans sibling `.m` files in the same directory and performs full cross-file body analysis:
- **External function registry**: `AnalysisContext.external_functions` maps function names to `ExternalSignature` (filename, param_count, return_count)
- **Priority order**: function_handle > KNOWN_BUILTINS > function_registry (same-file functions) > external_functions (workspace) > bound variable > W_UNKNOWN_FUNCTION
- **Filename-keyed dispatch**: MATLAB semantics require function name to match filename (e.g., `myFunc` must be in `myFunc.m`)
- **Cross-file body analysis**: External files are fully parsed and analyzed (lex -> parse -> analyze) to infer real return shapes (not just `unknown`)
- **Polymorphic caching**: Cross-file calls are cached per `(fname, arg_shapes)` to avoid redundant re-analysis
- **Dimension aliasing**: Symbolic dimension names propagate across file boundaries
- **Subfunction support**: Local functions in external files work correctly during cross-file analysis
- **Cycle detection**: A->B->A cross-file cycles are detected and return `unknown` gracefully (no infinite recursion)
- **`W_EXTERNAL_PARSE_ERROR`**: Emitted when an external `.m` file cannot be parsed; external body warnings are suppressed (not propagated to caller)
- **Content-addressed cache**: External file parse results are keyed by MD5 hash of file content; stale entries are invalidated when the file changes rather than on every save
- **Cross-file classdef resolution** (`Workspace.fs`, `Context.fs`): `scanWorkspace` returns a `(funcMap, classdefMap)` tuple; classdef files are detected via regex and routed to `externalClassdefs`; `loadExternalClassdef` parses the file and extracts `ClassInfo` plus method signatures; `tryLoadExternalClassdef` lazy-loads into the `classRegistry` at the first constructor reference; constructor and method dispatch both resolve cross-file the same way they do for same-file classdefs
- **Integrated everywhere**: Workspace scanning happens in CLI (`Cli.fs`), LSP server (`LspServer.fs`), and test runner (`TestRunner.fs`)

### User-Defined Functions and Lambdas

**Named functions** are defined with `function` keyword in 3 forms:
- Single return: `function result = name(params)`
- Multi-return: `function [out1, out2] = name(params)`
- Procedure: `function name(params)` (no return values)
- Nested: `function...end` blocks inside another function body

**`nargin`/`nargout` semantics**:
- `nargin` and `nargout` are not in `KNOWN_BUILTINS`; instead, they are injected into the function's initial environment as concrete singleton intervals at each call site (`nargin = [argc, argc]`, `nargout = [outc, outc]`)
- Calling with fewer arguments than declared is treated as optional (missing args start as bottom); calling with more than declared is still an error
- The cache key includes the argument count, so optional-arg functions called with different argc values are analyzed separately
- Interval refinement inside the body can then prune dead branches: `if nargin < 2` with `nargin = [2, 2]` has a provably false condition, so the true branch is effectively dead

**`varargin`/`varargout` semantics**:
- When the last declared parameter is `varargin`, extra call arguments beyond the named ones are bundled into a `Cell` with per-element shape tracking; `varargin{1}` returns the actual shape of the first extra argument via the existing `CurlyApply` element-map lookup
- The arg-count warning is suppressed when `hasVarargin` is detected on the callee
- `varargout`: extra output targets beyond the named return variables receive `UnknownShape`

**`global`/`persistent` semantics**:
- `globalStore: Dictionary<string, Shape>` on `CallContext` holds live global values; it is NOT saved or restored by `SnapshotScope` so changes persist across function analysis boundaries
- `globalDeclaredVars: HashSet<string>` on `ConstraintContext` IS snapshotted so the declaration set can be restored after branch joins
- `global x`: reads from `globalStore` if present (else `Bottom`); at function exit, the current value of `x` is written back to `globalStore`
- `persistent x`: binds as `Bottom` initially; the `if isempty(x), x = init; end` idiom resolves via `join(Bottom, shape) = shape`
- Functions that declare globals skip the analysis cache (their results are side-effecting and not safe to replay)

**Nested function semantics**:
- Nested functions are visible only within their enclosing function (not at top-level script scope)
- They have read access to parent workspace variables via `ScopedEnv` parent-pointer scope chains
- After a nested call returns, modified parent-scope variables are written back (post-call flush)
- Nested function parameters shadow parent variables without triggering write-back
- Sibling nested functions can call each other via forward reference pre-scan
- Dimension aliasing propagates through nested function boundaries

**Anonymous functions** (lambdas):
- Syntax: `@(params) expr` or `@(params) stmt` (single expression/statement body)
- Named handles: `@funcName` creates a handle to a named function or builtin

**Shared analysis semantics**:
- **Two-pass analysis**: Pass 1 registers all function definitions, Pass 2 analyzes script statements
- **Interprocedural analysis**: Functions and lambdas are analyzed at each call site with caller's argument shapes
- **Polymorphic caching**: Analysis results cached per (func_name, arg_shapes) or (lambda_id, arg_shapes) to avoid redundant re-analysis
- **Warning replay**: Cached warnings replayed with current call-site line number on cache hit
- **Dimension aliasing**: Symbolic dimension names propagate across boundaries (e.g., `f = @(n) zeros(n,n); B = f(5)` infers `B = matrix[5 x 5]`)
- **Recursion guard**: Recursive function calls emit `W_RECURSIVE_FUNCTION`; self-referencing lambdas emit `W_RECURSIVE_LAMBDA`
- **Dual-location warnings**: Warnings in bodies show both call site line and body line
- **Destructuring assignment**: `[a, b] = func(x)` binds multiple return values to variables
- **Return statement**: `return` keyword exits function early (MATLAB return has no value)
- **Catch-at-boundary**: EarlyReturn caught in If (non-returned branch used), loops (stop iteration), program (stop script)

**Lambda-specific semantics**:
- **Closure capture**: Lambdas capture environment by-value at definition time (MATLAB semantics)
- **Control flow join precision**: When branches assign different lambdas, both bodies analyzed and results joined at call site
- **Argument count validation**: Lambda calls emit `W_LAMBDA_ARG_COUNT_MISMATCH` if arg count doesn't match param count

### Best-Effort Analysis

When a definite mismatch is detected (e.g., inner dimension mismatch in `A*B`), the analyzer:
1. Emits a warning
2. Treats the result as `unknown`
3. Continues analysis to provide maximum information

**`cellfun`/`arrayfun` dispatch** (`EvalBuiltins.fs`):
- `resolveHandleOutputShape` synthesizes an `Apply` node and evaluates it through `evalExprFn` with the handle's argument shapes to determine per-element output shape
- `detectUniformOutput` scans the argument list for the `'UniformOutput'` name-value pair; when `false`, `cellfun` returns a cell matching the input cell dimensions and emits `W_CELLFUN_NON_UNIFORM` (strict-only)
- When `UniformOutput` is `true` (the default), a scalar per-element result produces a matrix matching the cell/matrix dimensions; named handles (`@func`) and lambdas (`@(x) expr`) both go through normal handle dispatch
- `arrayfun` follows the same logic applied to matrix inputs rather than cell inputs

**`classdef` support** (`Parser.fs`, `ClassInfo` registry, `Context.fs`):
- `ParseClassdef` extracts property names from `properties` blocks (bare names and `name = default` forms) and method `FunctionDef` nodes from `methods` blocks into a `ClassInfo` record
- The `ClassInfo` registry maps class names to their `ClassInfo`; when a call site matches a registered class name, the constructor body is analyzed and the result is a struct carrying the declared property fields
- Superclass syntax `classdef Foo < Bar` is parsed: the `< Bar` part is consumed and stored but the superclass is not resolved or analyzed
- **Method dispatch**: `classBindings: Dictionary<string, string>` on `CallContext` maps variable names to class names; constructor calls populate `classBindings`; when `evalApply` sees `obj.method(args)` and `obj` is in `classBindings`, it looks up the class in the `ClassInfo` registry and dispatches the call as `method(obj, args)` through the normal builtin dispatch path

## Known Behaviors and Gotchas

- Test discovery is dynamic via glob in `TestRunner.fs`
- `--strict` mode shows all warnings including low-confidence diagnostics; default mode suppresses strict-only codes
- `--license KEY` enables the pro-tier codes (`PRO_ONLY_CODES` in `Diagnostics.fs`) via Ed25519-signed license key validation; without a valid license, the CLI prints an upsell count and the LSP silently filters them; tests receive unfiltered warnings regardless of license status
- `--coder` runs a post-analysis pass that checks for MATLAB Coder incompatibilities; all six `W_CODER_*` codes are strict-only, so `--coder` without `--strict` will emit nothing; the Coder pass does not change shape inference, it only adds a compatibility scan on top; tests in `tests/coder/` use a `% MODE: coder` directive and the TestRunner enables the pass automatically for those files
- `--help` prints usage and exits 0; `--version` prints `conformal 2.8.0` and exits 0
- Struct field output is sorted alphabetically in both `shapeToString` and `printEnv`, making CLI output deterministic
- `% SKIP_TEST` in a test file causes the test runner to skip it silently (no pass, no fail)
- Indexed assignment (`M(i,j) = val`) does not check bounds because MATLAB auto-expands arrays on write
- Empty matrix `[]` (`matrix[0 x 0]`) is the identity element for concatenation: `[[] x]` produces `x`
- When editing the parser, check delimiter syncing and token precedence carefully (the parser emits IR nodes directly, so changes there affect both parse and IR structure)
- User-defined functions and lambdas are analyzed per-call-site (context-sensitive), not once globally
- Lambda body analysis uses caller's environment for closure resolution (by-value capture at definition time)
- Chained struct-index-struct assignment (`s.field(i).field2 = expr`) is parsed as `StructAssign` with flattened fields via a speculative parse with backtrack in the DOT-chain branch
- Colon visibility (`colon_visible: bool`) is threaded as a parameter through `parse_expr`/`parse_expr_rest` rather than mutating precedence; this is what allows `A([1:3])` to parse correctly with a colon inside the matrix literal index arg
- Matrix and cell literals call `parse_postfix` on the result, so `[1 2; 3 4]'` (transpose of a literal) and `{1, 2}'` parse without special-casing
- Open struct behavior: `_update_struct_field` creates an open struct from an unknown base instead of returning unknown; `FieldAccess` on an open struct with a missing field returns unknown silently
- Witness generation bails out conservatively: unknown dims produce no witness, quadratic+ symbolic terms produce no witness, and more than 8 free variables produce no witness; all returned witnesses are verified (dim_a_concrete != dim_b_concrete under the assignment)
- `#if FABLE_COMPILER` guards exist in `Lexer.fs` (regex compat) and `Workspace.fs` (System.IO) for Fable compilation
