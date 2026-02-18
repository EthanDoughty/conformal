# Conformal: MATLAB Shape Analyzer

Static shape and dimension analysis for MATLAB. No MATLAB license required.

Conformal finds matrix dimension errors before you run your code. If `A` is `3x4` and `B` is `5x2`, it can tell you that `A * B` has an inner dimension mismatch, and that `[A; B]` has mismatched column counts. It tracks symbolic dimensions like `n`, `m`, and `n+m` through assignments, function calls, control flow, and across sibling `.m` files in the same directory.

## What it catches

Most of what it catches comes down to dimension mismatches, whether that's in multiplication, concatenation, or element-wise operations. It can also flag type errors when you use structs or cells where numbers are expected, and it checks for index out of bounds, division by zero, and negative dimensions when it can prove them from the code. It follows shapes through user-defined functions, anonymous functions with closure capture, and cross-file calls to sibling `.m` files.

## In the editor

Diagnostics show up as underlines as you type, and you can hover any variable to see its inferred shape (`matrix[3 x n]`). If you use `*` where `.*` was probably intended, or `&&` where `&` should be, it suggests the fix. Function definitions appear in the sidebar and breadcrumbs, and the status bar shows warning counts. The extension registers its own MATLAB grammar, so you don't need the MathWorks extension.

The analyzer handles most files in under 100ms, so real-time analysis is on by default with a 500ms debounce. A 700-line file with 36 warnings takes about 99ms, and cross-file workspace analysis runs in under 70ms. There's no MATLAB runtime involved.

## Install

Search "Conformal" in VS Code Extensions, or run:

```bash
code --install-extension EthanDoughty.conformal
```

On first activation, the extension creates a Python venv, installs its dependencies, and uses the bundled analyzer. You shouldn't need to configure anything.

Requires Python 3.10+.

## How it works

The extension runs `python3 -m lsp` as an LSP subprocess over stdio. The server analyzes your `.m` files and publishes diagnostics back to the editor. When you save a file, it re-analyzes siblings that might depend on it. If the server crashes, it can auto-recover up to 3 times.

Under the hood, there are 128 builtin shape rules, symbolic dimension tracking, constraint solving, interval analysis, and fixed-point loop convergence, all validated by 270 tests across 15 categories.

## Commands

| Command | What it does |
|---------|-------------|
| `Conformal: Analyze Current File` | Save and re-analyze |
| `Conformal: Toggle Fixpoint Mode` | Fixed-point loop analysis |
| `Conformal: Toggle Strict Mode` | Fail on unsupported constructs |
| `Conformal: Restart Server` | Restart the LSP server |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.pythonPath` | `python3` | Leave default for auto-setup |
| `conformal.serverPath` | _(empty)_ | Dev use only |
| `conformal.fixpoint` | `false` | Fixed-point loop analysis |
| `conformal.strict` | `false` | Strict mode |
| `conformal.analyzeOnChange` | `true` | Keystroke analysis, 500ms debounce |

[VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal) · [GitHub](https://github.com/EthanDoughty/conformal)
