# Conformal — MATLAB Shape Analyzer

Static shape and dimension analysis for MATLAB, right in your editor.

Conformal catches matrix dimension errors before you run your code: inner dimension mismatches in `A * B`, incompatible concatenation in `[A; B]`, and more.

## Features

- **Real-time diagnostics** — dimension mismatches appear as squiggly underlines on save
- **Hover tooltips** — see inferred shapes for any variable (e.g. `matrix[3 x n]`)
- **Symbolic tracking** — dimensions like `n`, `m`, `n+m` are tracked through operations
- **162 analysis rules** — builtins, control flow, functions, lambdas, cells, structs, constraints

## Install

```bash
pip install -e '.[lsp]'            # from the conformal repo root
```

Then install this extension (`.vsix`) via the Extensions panel or:

```bash
code --install-extension conformal-1.0.0.vsix
```

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `conformal.pythonPath` | `python3` | Python interpreter (must have pygls installed) |
| `conformal.serverPath` | _(empty)_ | Path to repo root (only needed if not pip-installed) |
| `conformal.fixpoint` | `false` | Enable fixed-point loop analysis |
| `conformal.strict` | `false` | Fail on unsupported constructs |
| `conformal.analyzeOnChange` | `false` | Analyze on every keystroke (with 500ms debounce) |

## How It Works

The extension spawns `python3 -m lsp` as a Language Server Protocol subprocess. The LSP server runs the Conformal analyzer on your MATLAB files and publishes diagnostics back to VS Code.

## More Information

- [GitHub Repository](https://github.com/EthanDoughty/conformal)
- [Full Documentation](https://github.com/EthanDoughty/conformal#readme)
