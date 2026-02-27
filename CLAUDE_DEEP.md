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

**Strict / fixpoint / witness modes**:
```bash
cd src && dotnet run -- --strict ../tests/recovery/struct_field.m
cd src && dotnet run -- --fixpoint ../tests/loops/matrix_growth.m
cd src && dotnet run -- --witness ../tests/basics/inner_dim_mismatch.m
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

### F# Analyzer (`src/`) — 36 files, ~11.5K lines

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
- `Constraints.fs`: Dimension constraint tracking
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

### VS Code Extension (`vscode-conformal/`) — v2.0.0, Fable-powered

- Runs F#-compiled-to-JavaScript analyzer in-process via Node.js module transport (no external dependencies)
- `src/extension.ts`: LanguageClient with TransportKind.ipc, status bar, commands
- `src/server.ts`: LSP server using `vscode-languageserver` + Fable-compiled analyzer (`require('./fable-out/Interop.js')`)
- `esbuild.mjs`: Bundles client + server into `out/extension.js` and `out/server.js`
- Configuration settings: fixpoint, strict, analyzeOnChange
- Commands: analyzeFile, toggleFixpoint, toggleStrict, restartServer
- 181KB VSIX; registers `.m` language with TextMate grammar
- Auto-reclassifies `.m` files from Objective-C to MATLAB via `firstLine` + `reclassifyIfMatlab()`

### Diagnostic System

All warnings are structured as `Diagnostic` record instances:
- **Fields**: `line`, `col`, `code` (W_* prefix), `message`, optional `related_line`, `related_col`
- **Warning codes**: All warnings have stable W_* codes
- **Warning tiers**: `STRICT_ONLY_CODES` in `Diagnostics.fs` defines codes suppressed in default mode (shown only with `--strict`). Filtering is a post-analysis presentation concern applied in CLI and LSP. Tests receive unfiltered warnings.
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
- **Interval analysis**: Integer interval domain `[lo, hi]` tracks scalar value ranges in parallel with shape inference; bounds accept `SymDim` values for symbolic upper bounds (`for i = 1:n` -> `i in [1, n]`, symbolic bound); condition refinement narrows intervals inside branch bodies (`if x > 0` refines `x` to `[1, +inf]`), eliminating false-positive OOB/negative-dim warnings when guards prove safety; intervals join conservatively across branches; enables `W_DIVISION_BY_ZERO`, `W_INDEX_OUT_OF_BOUNDS`, and `W_POSSIBLY_NEGATIVE_DIM` checks
- **Range-valued dimensions**: `Range(lo, hi)` dimension variant for interval-valued shape dimensions, used in conditional accumulation where the final dimension depends on which branches execute

## Test File Format

Tests use inline assertions in MATLAB comments:
```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]   (override when --fixpoint active)
```

The test runner (`TestRunner.fs`) validates these expectations against analysis results. Test files are organized in `tests/` subdirectories by category (18 categories, 370 tests) and discovered dynamically via glob. Run `cd src && dotnet run -- --tests` to see the current count.

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
- **Integrated everywhere**: Workspace scanning happens in CLI (`Cli.fs`), LSP server (`LspServer.fs`), and test runner (`TestRunner.fs`)

### User-Defined Functions and Lambdas

**Named functions** are defined with `function` keyword in 3 forms:
- Single return: `function result = name(params)`
- Multi-return: `function [out1, out2] = name(params)`
- Procedure: `function name(params)` (no return values)
- Nested: `function...end` blocks inside another function body

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

## Known Behaviors and Gotchas

- Test discovery is dynamic via glob in `TestRunner.fs`
- `--strict` mode shows all warnings including low-confidence diagnostics; default mode suppresses strict-only codes
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
