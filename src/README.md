# Conformal: F# Source

This directory contains the F# implementation across four projects. All new features go here first.

## Project layout

```
src/
  core/       27 files — ConformalCore library (zero NuGet deps, AOT-compatible)
  shared/      1 file  — ConformalShared library (License.fs, NSec.Cryptography)
  analyzer/   11 files — ConformalAnalyzer executable (CLI, LSP, tests)
  migrate/     6 files — ConformalMigrate executable (MATLAB-to-Python transpiler)
```

## Build and run

```bash
# Build the full solution
dotnet build Conformal.sln

# Run the full test suite (502 tests across 22 categories + 28 property-based tests)
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --tests

# Run migrate tests (27 paired .m/.py tests)
dotnet run --project src/migrate/ConformalMigrate.fsproj -- --test-migrate

# Analyze a single file
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- tests/basics/inner_dim_mismatch.m

# Strict mode (all 11 low-confidence codes shown)
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --strict tests/recovery/struct_field.m

# Fixed-point loop analysis
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --fixpoint tests/loops/matrix_growth.m

# Incorrectness witness generation
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --witness tests/basics/inner_dim_mismatch.m

# Start the native .NET LSP server
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --lsp

# Translate a MATLAB file to Python
dotnet run --project src/migrate/ConformalMigrate.fsproj -- input.m --stdout
```

Dependencies: .NET 8, FsCheck (property tests), Ionide.LanguageServerProtocol (LSP).

## Source files

### core/ — shape analysis library

| File | Role |
|------|------|
| `Ir.fs` | Discriminated unions for `Expr` and `Stmt` (the IR) |
| `Lexer.fs` | Tokenizer; `#if FABLE_COMPILER` guards for JS compatibility |
| `Parser.fs` | Recursive-descent parser, emits typed IR nodes directly |
| `Shapes.fs` | Shape domain: `Scalar`, `Matrix`, `String`, `Struct`, `FunctionHandle`, `Cell`, `Unknown`, `Bottom` |
| `SymDim.fs` | Symbolic dimension polynomials with rational coefficients |
| `SharedTypes.fs` | Shared types used across modules (Interval, FunctionSignature, etc.) |
| `WarningCodes.fs` | `WarningCode` discriminated union and serialization |
| `Diagnostics.fs` | Tier classification (strict/pro/coder), diagnostic records, message builders |
| `Env.fs` | Variable environment with parent-pointer scope chains |
| `EvalExpr.fs` | Expression evaluation and indexing logic |
| `EvalBinop.fs` | Binary operator shape rules |
| `EvalBuiltins.fs` | Builtin dispatch: 635 recognized, 315 with shape rules |
| `StmtFuncAnalysis.fs` | Statement dispatch, function and lambda analysis |
| `Analysis.fs` | Entry point, two-pass analysis |
| `AnalysisCore.fs` | Shared compatibility checks |
| `Context.fs` | `AnalysisContext` with call, constraint, and workspace sub-objects |
| `Builtins.fs` | Builtin name catalog |
| `Constraints.fs` | Dimension constraint tracking and path-sensitive joins |
| `Intervals.fs` | Integer interval domain with symbolic bounds |
| `DimEquiv.fs` | Dimension equivalence via union-find |
| `DimExtract.fs` | Dimension extraction from expressions |
| `EndHelpers.fs` | `end` keyword resolution |
| `MatrixLiterals.fs` | Matrix literal shape inference |
| `PathConstraints.fs` | Branch-aware path constraint stack |
| `Witness.fs` | Incorrectness witness generation |
| `Workspace.fs` | Cross-file workspace scanning and analysis |
| `Json.fs` | JSON serialization for diagnostics |

### shared/ — shared utilities

| File | Role |
|------|------|
| `License.fs` | License key validation (reserved for future Migrate licensing) |

### analyzer/ — CLI, LSP, and tests

| File | Role |
|------|------|
| `Cli.fs` | CLI argument parsing and dispatch |
| `Program.fs` | Entry point |
| `LspServer.fs` | Native .NET LSP server (Ionide.LanguageServerProtocol) |
| `LspDiagnostics.fs` | LSP diagnostic conversion and severity mapping |
| `LspHover.fs` | Hover provider (inferred shapes, function signatures) |
| `LspSymbols.fs` | Document symbol provider (breadcrumbs, outline) |
| `LspDefinition.fs` | Go-to-definition provider |
| `LspCodeActions.fs` | Quick-fix code actions (`*` -> `.*`, `&&` -> `&`, etc.) |
| `LspInlayHints.fs` | Inlay hints: shape annotations on first assignment |
| `PropertyTests.fs` | 28 FsCheck property-based tests across 6 lattice sections |
| `TestRunner.fs` | Test runner for `.m` inline assertion tests |

### migrate/ — MATLAB-to-Python transpiler

| File | Role |
|------|------|
| `PyAst.fs` | Python AST types |
| `Translate.fs` | MATLAB IR to Python AST translation |
| `BuiltinMap.fs` | 162 builtin function mappings with 26 ArgTransform variants |
| `CopySemantics.fs` | Value vs reference semantics for Python output |
| `Emit.fs` | Python source code emitter |
| `Program.fs` | CLI entry point and test runner |

## Fable compilation

The `vscode-conformal/fable/` directory contains a separate Fable project that compiles the core `.fs` files to JavaScript for use inside the VS Code extension. Files that use .NET-specific APIs (LSP, file I/O, CLI) are excluded; `Interop.fs` exposes a TypeScript-callable `analyzeSource` function.

```bash
cd src && dotnet fable ../vscode-conformal/fable/ConformalFable.fsproj --outDir ../vscode-conformal/src/fable-out
cd ../vscode-conformal && node esbuild.mjs
```

## Test format

Each `.m` test file embeds its expected behavior as inline assertions:

```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]
```

Tests are discovered dynamically from `tests/**/*.m`. Run `dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- --tests` to see the current pass/fail count.
