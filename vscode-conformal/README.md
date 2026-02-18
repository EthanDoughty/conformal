# Conformal — MATLAB Shape Analyzer

Static shape and dimension analysis for MATLAB, right in your editor. **No MATLAB license required.**

Conformal catches matrix dimension errors before you run your code — inner dimension mismatches in `A * B`, incompatible concatenation in `[A; B]`, type errors in arithmetic, and more. It tracks symbolic dimensions like `n` and `m` through operations, functions, and control flow.

## What It Catches

- **Dimension mismatches** — `A * B` when inner dims conflict, `[A; B]` with different column counts
- **Type errors** — arithmetic on strings/structs, transposing non-numeric types
- **Index out of bounds** — `A(5, :)` when row count is 3
- **Division by zero** — `x / y` when `y` is provably zero
- **Negative dimensions** — `zeros(n-5, m)` when `n < 5`
- **Shape-through-functions** — tracks dimensions across user-defined and anonymous functions
- **Cross-file inference** — resolves shapes from sibling `.m` files in the same directory

## Editor Features

- **Real-time diagnostics** — dimension errors appear as squiggly underlines on save
- **Hover tooltips** — see inferred shapes for variables (`matrix[3 x n]`), builtins, and functions
- **Document outline** — function definitions in the Explorer sidebar and breadcrumbs
- **Quick fixes** — `*` → `.*`, `&&` → `&`, `||` → `|`
- **Status bar** — error/warning counts and active modes (fixpoint, strict)
- **Syntax highlighting** — built-in MATLAB grammar (no MathWorks extension needed)
- **Auto-restart** — server recovers from crashes automatically (up to 3 retries)

## Analysis Capabilities

- 270 analysis tests across 15 categories
- 128 built-in function shape rules (zeros, ones, reshape, kron, blkdiag, ...)
- Symbolic dimension tracking (`n`, `m`, `n+m`, `2*n`)
- Constraint solving with conflict detection
- Interval/value range analysis for bounds checking
- Fixed-point loop convergence with accumulation refinement
- Polymorphic function caching (per argument shapes)

## Install

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal) — search "Conformal" in Extensions.

The LSP server requires Python 3.10+ with pygls:
```bash
pip install 'pygls>=2.0'
```

## Commands

| Command | Description |
|---------|-------------|
| `Conformal: Analyze Current File` | Save and re-analyze the active `.m` file |
| `Conformal: Toggle Fixpoint Mode` | Enable/disable fixed-point loop analysis |
| `Conformal: Toggle Strict Mode` | Fail on unsupported constructs |
| `Conformal: Restart Server` | Restart the LSP server |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.pythonPath` | `python3` | Python interpreter (must have pygls) |
| `conformal.serverPath` | _(empty)_ | Path to repo root (if not pip-installed) |
| `conformal.fixpoint` | `false` | Fixed-point loop analysis |
| `conformal.strict` | `false` | Fail on unsupported constructs |
| `conformal.analyzeOnChange` | `false` | Analyze on keystrokes (500ms debounce) |

## How It Works

The extension spawns `python3 -m lsp` as a Language Server Protocol subprocess. The server runs the Conformal analyzer on your `.m` files and publishes diagnostics. Hover requests show shapes from the last successful analysis. Saving a file triggers re-analysis of sibling files that may depend on it.

Catches dimension bugs that MATLAB's built-in linter misses.

## Links

- [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=EthanDoughty.conformal)
- [GitHub Repository](https://github.com/EthanDoughty/conformal)
- [Full Documentation](https://github.com/EthanDoughty/conformal#readme)
