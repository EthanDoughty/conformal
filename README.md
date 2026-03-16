<div align="center">

# Conformal

### Static Shape & Dimension Analysis for MATLAB

[![Version](https://img.shields.io/badge/version-3.4.0-orange.svg)](#cli-options)
[![VS Code](https://img.shields.io/badge/VS%20Code-Marketplace-007ACC.svg)](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4.svg)](https://dotnet.microsoft.com/download)
[![Tests](https://img.shields.io/badge/tests-515%20passing-brightgreen.svg)](#test-suite)
[![License](https://img.shields.io/badge/license-BSL--1.1-purple.svg)](LICENSE)

*Matrices must be **conformable** before they can perform. Conformal makes sure they are.*

> Conformal is an independent project and is not affiliated with, endorsed by, or connected to MathWorks, Inc. MATLAB is a registered trademark of MathWorks, Inc.

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

## Screenshots

Inline diagnostics flag dimension mismatches as you type:

![Inline diagnostics](vscode-conformal/images/Conformal_Example_1.png)

Hover any variable to see its inferred shape:

![Hover shape](vscode-conformal/images/Conformal_Example_2.png)

Inlay hints show shapes on first assignment:

![Inlay hints](vscode-conformal/images/Conformal_Example_3.png)

## Quick Start

**VS Code** (the recommended option): Install Conformal from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal) by searching "Conformal" in Extensions, or run the following command:
```bash
code --install-extension EthanDoughty.conformal
```
Open any `.m` file and the diagnostics appear as underlines. Hover a variable to see its inferred shape. No configuration is needed.

**CLI**: Requires [.NET 8.0 SDK](https://dotnet.microsoft.com/download) or later. Again, no MATLAB installation is required.
```bash
git clone https://github.com/EthanDoughty/conformal.git
cd conformal
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- tests/basics/inner_dim_mismatch.m
```

## What Conformal Catches

Conformal tracks the shape of every variable through your program and flags dimension mismatches, type errors, and structural problems. Here's what it covers:

**Arithmetic and operators.** Conformal detects dimension mismatches in `+`, `-`, `*`, `.*`, `./`, `^`, `.^`, and backslash `\`. Scalar-matrix broadcasting is handled, so `s * A` works without a warning. When you use `*` where `.*` was probably intended, Conformal suggests the fix. Comparison operators return the broadcast shape of their operands, so logical indexing like `A(A > 0)` can be tracked correctly.

**Concatenation and literals.** Horizontal concatenation `[A B]` checks that row counts match, and vertical `[A; B]` checks columns. Symbolic dimensions compose through concatenation, so `[A B]` where A is `n x k` and B is `n x m` gives `n x (k+m)`. Matrix literal spacing is handled correctly, so `[1 -2; 3 4]` parses as four elements rather than as subtraction.

**Indexing.** Parenthesized indexing `A(i,j)`, slice indexing `A(:,j)`, range indexing `A(2:5,:)`, curly-brace cell indexing `C{i}`, and the `end` keyword with arithmetic (`A(end-1, :)`) are all supported. Indexed assignment preserves the matrix shape, and read-side out-of-bounds checking is available with a Pro license.

**Functions.** Over 635 MATLAB builtins are recognized, and around 315 have explicit shape rules. Matrix constructors (`zeros`, `ones`, `eye`, `rand`), reductions (`sum`, `mean`, `max` with dimension args), reshaping, type predicates, and linear algebra functions are all covered. User-defined functions are analyzed at each call site with the caller's argument shapes. Nested functions, anonymous functions, `nargin`/`nargout`, `varargin`/`varargout`, and cross-file workspace resolution are all supported.

**Symbolic dimensions.** Variables with unknown concrete size get symbolic names like `n`, `m`, `k`, and those names propagate through operations. Symbolic dimensions use a polynomial representation with rational coefficients, so `n+m` and `m+n` are recognized as equal, and `n+n` simplifies to `2*n`.

**Data structures.** Conformal tracks struct fields, cell array elements, and basic classdef objects. Missing field access on a known struct emits a warning. Cell arrays support per-element shape tracking with literal indexing.

**Control flow.** `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`, `try`/`catch`, `break`, `continue`, and `return` are all supported. When branches assign different shapes to the same variable, Conformal joins them conservatively. Loops can use single-pass analysis or widening-based fixpoint iteration via `--fixpoint`.

**Interval analysis.** In parallel with shape inference, Conformal tracks scalar integer variables through an interval domain. This enables out-of-bounds indexing detection, division-by-zero detection, and negative dimension warnings. Branch conditions narrow intervals, and a Pentagon relational domain tracks upper-bound relations from loop ranges to suppress false positives.

**Witness generation.** For dimension conflict warnings, Conformal can produce a concrete counterexample proving the bug is real. In `--witness filter` mode, only warnings with a verified witness are shown, which gives you zero false positives.

**MATLAB Coder compatibility.** The `--coder` flag adds a post-analysis pass that checks for constructs MATLAB Coder can't handle, including variable-size arrays, cell arrays, dynamic field access, try/catch, unsupported builtins, and recursion.

## VS Code Extension

The VS Code extension runs the analyzer in-process, since the F# codebase is compiled to JavaScript using the Fable tool. There is no external runtime dependency: no Python, no .NET, no subprocess. The compiled analyzer is bundled directly into the extension.

Diagnostics appear as underlines as you type, with a configurable 500ms debounce. You can hover any variable to see its inferred shape. Go-to-definition works for user-defined and cross-file functions. Function definitions show in the sidebar via document symbols. The extension includes built-in MATLAB syntax highlighting, so you don't need the MathWorks extension.

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.fixpoint` | `false` | Enable fixed-point loop analysis |
| `conformal.strict` | `false` | Show all warnings including low-confidence diagnostics |
| `conformal.analyzeOnChange` | `true` | Analyze as you type (500ms debounce) |
| `conformal.inlayHints` | `true` | Show inferred shapes as inlay hints on first assignment |

## CLI Options

```bash
dotnet run --project src/analyzer/ConformalAnalyzer.fsproj -- file.m
```

| Flag | What it does |
|------|-------------|
| `--tests` | Run the full test suite (515 tests across 22 categories) |
| `--strict` | Show all warnings including informational and low-confidence diagnostics |
| `--fixpoint` | Use widening-based fixpoint iteration for loop analysis |
| `--witness [MODE]` | Attach incorrectness witnesses (`enrich`, `filter`, or `tag`) |
| `--coder` | Run the MATLAB Coder compatibility pass (combine with `--strict`) |
| `--quiet` | Suppress per-test output during `--tests`, only print failures |
| `--lsp` | Start the native Language Server Protocol server |
| `--version` | Print version and exit |

The exit code is `0` on success and `1` on a parse error or test failure.

## Performance

The single-file analysis takes under 100ms, even for 700-line files with dozens of warnings. The full test suite finishes in about one second. The VS Code extension runs the analyzer on every keystroke with a 500ms debounce, since it is compiled to JavaScript and runs in-process with no subprocess startup cost.

## Real-World Compatibility

To check how Conformal holds up on real MATLAB code, a corpus of 1,197 `.m` files was drawn from 11 open-source repos on GitHub, covering robotics, signal processing, scientific computing, and computer vision. In default mode, the corpus produces zero crashes and zero false positives.

The repos include gptoolbox, vlfeat, prmlt, petercorke/robotics-toolbox-matlab, rpng/kalibr_allan, and others. These files use a wide range of MATLAB idioms: pre-2016 end-less function definitions, space-separated multi-return syntax, Latin-1 encoded files, `\` for linear solves, and complex matrix literal spacing. Parser robustness improvements came directly from failures on this corpus.

## Warning Tiers

By default, Conformal shows all high-confidence warnings, including shape errors, type errors, indexing checks, interval-based checks like `W_INDEX_OUT_OF_BOUNDS` and `W_DIVISION_BY_ZERO`, constraint conflicts, and cross-file resolution. All 36 default codes are available with no configuration required.

The `--strict` flag adds 11 lower-confidence codes like `W_SUSPICIOUS_COMPARISON` and `W_REASSIGN_INCOMPATIBLE`, so you can run default mode in CI without false-positive noise and use strict mode when you want a fuller picture.

## Conformal Migrate (Preview)

Conformal also includes a MATLAB-to-Python transpiler that uses the shape analysis to make better translation decisions than a purely syntactic tool can. It handles 168 MATLAB builtins, 1-to-0 index conversion with constant folding, `varargin` to `*args`, copy semantics, and shape-aware operator dispatch (for example, using `np.dot` for matrix multiply and `*` for element-wise). The transpiler is a separate tool and is under active development.

## Test Suite

Conformal is validated by 515 self-checking MATLAB programs organized into 22 categories, plus 28 property-based lattice tests via FsCheck. Each test file embeds its expected behavior as inline assertions (`% EXPECT: A = matrix[3 x 4]`, `% EXPECT_WARNING: W_INNER_DIM_MISMATCH`), and the test runner checks that Conformal's output matches.

For the full test listing, see [docs/tests.md](docs/tests.md).

## Project Structure

```
src/core/               F# core library (lexer, parser, shape inference, builtins, diagnostics)
src/shared/             Shared utilities
src/analyzer/           CLI, LSP server, test runner
src/migrate/            MATLAB-to-Python transpiler (~1,700 LOC)
vscode-conformal/       VS Code extension (TypeScript client + Fable-compiled analyzer)
  fable/                Fable compilation project (F# to JavaScript, shares core .fs files)
  src/                  TypeScript extension and LSP server code
tests/                  515 self-checking MATLAB programs in 22 categories
.github/                CI workflow (build, test, compile Fable, package VSIX)
```

## Limitations

Conformal analyzes a subset of MATLAB. It focuses on the matrix-heavy computational core where dimension errors are most common and most costly. There is no support for `eval` or `str2func`, no N-D arrays beyond 2-D matrices, no complex number tracking, and no cross-directory resolution with `addpath`. Basic `classdef` support is included (properties, constructor, method dispatch), but inheritance is not resolved. For the full details on what is and isn't covered, see [docs/analysis.md](docs/analysis.md).

<details>
<summary><h2>References</h2></summary>

Conformal's abstract interpretation techniques draw on decades of research in static analysis and formal methods.

### Foundational

- P. Cousot and R. Cousot, "Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints," *POPL*, 1977. [ACM DL](https://dl.acm.org/doi/10.1145/512950.512973)
- P. Cousot, R. Cousot, and L. Mauborgne, "The Reduced Product of Abstract Domains and the Combination of Decision Procedures," *FoSSaCS*, 2011.
- P. Cousot et al., "A Personal Historical Perspective on Abstract Interpretation," 2024.

### Abstract Domains

- A. Mine, "The Octagon Abstract Domain," *Higher-Order and Symbolic Computation*, vol. 19, no. 1, 2006. [HAL](https://hal.science/hal-00136639/document)
- F. Logozzo and M. Fahndrich, "Pentagons: A Weakly Relational Abstract Domain," *SAS*, 2008. [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/01/pentagons.pdf)
- F. Ranzato, "The Best of Abstract Interpretations," *POPL*, 2025.
- A. Pitchanathan et al., "Strided Difference Bound Matrices," *CAV*, 2024.
- A. Lesbre et al., "Relational Abstractions Based on Labeled Union-Find," *PLDI*, 2025.

### Industrial Analyzers

- B. Blanchet et al., "A Static Analyzer for Large Safety-Critical Software," *PLDI*, 2003. (Astree)
- P. Cousot et al., "The ASTREE Analyzer," *ESOP*, 2005. [PDF](https://www.di.ens.fr/~cousot/publications.www/CousotEtAl-ESOP05.pdf)

### Data Structures

- S. Conchon and J.-C. Filliatre, "A Persistent Union-Find Data Structure," *ML Workshop*, 2007.

### MATLAB-Specific

- P. Joisha and P. Banerjee, "Static Array Storage Optimization in MATLAB," *PLDI*, 2003.

### Constraint and Shape Inference

- G. Zilberstein and D. Dreyer, "A Combination of Abstract Interpretation and Constraint Programming," 2024. [PDF](https://ghilesz.github.io/papers/manuscrit.pdf)
- MLIR Shape Inference. [Documentation](https://mlir.llvm.org/docs/ShapeInference/)

### Safety Standards

- DO-178C, "Software Considerations in Airborne Systems and Equipment Certification," RTCA, 2011.
- IEC 61508, "Functional Safety of Electrical/Electronic/Programmable Electronic Safety-Related Systems."

</details>
