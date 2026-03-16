# Conformal: MATLAB Shape Analyzer

Static shape and dimension analysis for MATLAB. No MATLAB license required, and no Python, no .NET, nothing to install beyond the extension itself.

Conformal finds matrix dimension errors before you run your code. If `A` is `3x4` and `B` is `5x2`, it can tell you that `A * B` has an inner dimension mismatch, and that `[A; B]` has mismatched column counts. It follows shapes through assignments, function calls, control flow, and across sibling `.m` files in the same directory, tracking symbolic dimensions like `n`, `m`, and `n+m` the whole way.

## Screenshots

![Inline diagnostics](images/Conformal_Example_1.png)
*Dimension mismatches appear as underlines while you type.*

![Hover shape](images/Conformal_Example_2.png)
*Hover any variable to see its inferred shape.*

![Inlay hints](images/Conformal_Example_3.png)
*Inlay hints show shapes on first assignment of each variable.*

## What it catches

Most of what Conformal catches comes down to dimension mismatches, whether that's in multiplication, concatenation, element-wise operations, or backslash solves. It can also flag type errors when you use structs or cells where numbers are expected, and it checks for index out of bounds, division by zero, and negative dimensions when it can prove them from the code. It follows shapes through user-defined functions (including pre-2016 end-less definitions and no-arg procedures), anonymous functions with closure capture, and cross-file calls to sibling `.m` files.

By default, Conformal shows only high-confidence warnings. The `strict` setting adds 11 low-confidence codes like `W_SUSPICIOUS_COMPARISON`, mostly things that could be false positives.

## In the editor

First, diagnostics show up as underlines as you type, and you can hover any variable to see its inferred shape (like `matrix[3 x n]`). Go-to-definition works for user-defined functions and external functions in sibling files. Second, if you use `*` where `.*` was probably intended, or `&&` where `&` should be, Conformal suggests the fix. Function definitions appear in the sidebar and breadcrumbs via document symbols, and the status bar shows warning and error counts along with which modes are active.

Presently, Conformal handles most files in under 100ms, so real-time analysis is on by default with a 500ms debounce. A 700-line file with 36 warnings takes about 99ms, and cross-file workspace analysis runs in under 70ms. The extension also registers its own MATLAB grammar, so you don't need the MathWorks extension.

## Install

Search "Conformal" in VS Code Extensions, or run:

```bash
code --install-extension EthanDoughty.conformal
```

You shouldn't need to configure anything. The extension bundles the analyzer directly -- no subprocess, no Python, and no .NET required on your machine.

## Commands

| Command | What it does |
|---------|-------------|
| `Conformal: Analyze Current File` | Save and re-analyze the current file |
| `Conformal: Toggle Fixpoint Mode` | Enable or disable fixed-point loop analysis |
| `Conformal: Toggle Strict Mode` | Show all warnings including informational and low-confidence diagnostics |
| `Conformal: Restart Server` | Restart the LSP server |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.fixpoint` | `false` | Fixed-point loop analysis for convergence in loops |
| `conformal.strict` | `false` | Show all warnings including informational and low-confidence diagnostics |
| `conformal.analyzeOnChange` | `true` | Keystroke analysis with a 500ms debounce |
| `conformal.inlayHints` | `true` | Show inferred shapes as inlay hints on first assignment. If hints feel noisy, set to `false` |

[VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal) · [GitHub](https://github.com/EthanDoughty/conformal)
