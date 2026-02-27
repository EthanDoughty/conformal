# Conformal: F# Analyzer

This directory contains the F# implementation of the Conformal static shape analyzer. It is the authoritative implementation; all new features go here first.

## Build and run

```bash
# Run the full test suite (370 .m tests + 28 property-based tests)
dotnet run -- --tests

# Analyze a single file
dotnet run -- ../tests/basics/inner_dim_mismatch.m

# Strict mode (all 19 low-confidence codes shown)
dotnet run -- --strict ../tests/recovery/struct_field.m

# Fixed-point loop analysis
dotnet run -- --fixpoint ../tests/loops/matrix_growth.m

# Incorrectness witness generation
dotnet run -- --witness ../tests/basics/inner_dim_mismatch.m

# Start the native .NET LSP server
dotnet run -- --lsp
```

Build: `dotnet build`. The project targets .NET 8 and has no external dependencies besides FsCheck (for property tests) and Ionide.LanguageServerProtocol (for LSP).

## Source files

The analyzer is organized into a pipeline: lexer -> parser -> shape inference, with supporting modules for diagnostics, LSP, and testing.

| File | Role |
|------|------|
| `Ir.fs` | Discriminated unions for `Expr` and `Stmt` (the IR) |
| `Lexer.fs` | Tokenizer; `#if FABLE_COMPILER` guards for JS compatibility |
| `Parser.fs` | Recursive-descent parser, emits typed IR nodes directly |
| `Shapes.fs` | Shape domain: `Scalar`, `Matrix`, `String`, `Struct`, `FunctionHandle`, `Cell`, `Unknown`, `Bottom` |
| `SymDim.fs` | Symbolic dimension polynomials with rational coefficients |
| `SharedTypes.fs` | Shared types used across modules (Interval, FunctionSignature, etc.) |
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
| `DimExtract.fs` | Dimension extraction from expressions |
| `EndHelpers.fs` | `end` keyword resolution |
| `MatrixLiterals.fs` | Matrix literal shape inference |
| `PathConstraints.fs` | Branch-aware path constraint stack |
| `Witness.fs` | Incorrectness witness generation |
| `Workspace.fs` | Cross-file workspace scanning and analysis |
| `Diagnostics.fs` | Warning codes and strict-only classification |
| `Json.fs` | JSON serialization for diagnostics |
| `LspServer.fs` | Native .NET LSP server (Ionide.LanguageServerProtocol) |
| `LspDiagnostics.fs` | LSP diagnostic conversion and severity mapping |
| `LspHover.fs` | Hover provider (inferred shapes, function signatures) |
| `LspSymbols.fs` | Document symbol provider (breadcrumbs, outline) |
| `LspDefinition.fs` | Go-to-definition provider |
| `LspCodeActions.fs` | Quick-fix code actions (`*` -> `.*`, `&&` -> `&`, etc.) |
| `PropertyTests.fs` | 28 FsCheck property-based tests across 6 lattice sections |
| `TestRunner.fs` | Test runner for `.m` inline assertion tests |
| `Cli.fs` | CLI argument parsing and dispatch |
| `Program.fs` | Entry point |

## Fable compilation

The `vscode-conformal/fable/` directory contains a separate Fable project that compiles most of these `.fs` files to JavaScript for use inside the VS Code extension. Files that use .NET-specific APIs (LSP server, file I/O, CLI) are excluded; `Interop.fs` exposes a TypeScript-callable `analyzeSource` function. The compiled output lands in `vscode-conformal/src/fable-out/`.

```bash
cd ../vscode-conformal/fable && dotnet fable --outDir ../src/fable-out
cd .. && node esbuild.mjs
```

## Test format

Each `.m` test file embeds its expected output as inline assertions:

```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
% EXPECT_FIXPOINT: A = matrix[None x None]
```

Tests are discovered dynamically from `tests/**/*.m`. Run `dotnet run -- --tests` to see the current pass/fail count.
