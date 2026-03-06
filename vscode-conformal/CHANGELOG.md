# Changelog

## 3.2.0

- Fix spurious W_END_OUTSIDE_INDEXING from parenthesized `for(i = 1:n)` syntax
- Fix switch parsing when blank lines appear before first case
- Add `?ClassName` metaclass operator support (no longer crashes lexer)
- Fix string-operator collision: `'*'` inside `[T '*']` no longer misparses as multiply
- PARFOR added to endless-function block-opener detection
- Dogfood corpus expanded to 1197 files, 0 crashes, 0 false positives
- 469 tests

## 3.1.0

- Workspace files correctly shadow builtins (MATLAB dispatch order: local > workspace > builtin)
- Multi-return support for kmeans, intersect, union, setdiff, ismember
- Multiline matrix literal parsing fix (semicolons followed by newlines)
- Empty matrix [] now type-neutral in concatenation (no false W_CONCAT_TYPE_MISMATCH)
- Fewer targets than outputs is valid MATLAB (e.g. [a, b] = f() when f returns 3)
- Dogfood corpus expanded to 358 files (added 208-file vlfeat, 150-file prmlt)
- 465 tests

## 3.0.0

- Recursive workspace scanning (resolves cross-directory function calls up to 3 levels deep)
- feval/str2func dispatch (resolves string literal and function handle arguments)
- private/ directory function resolution
- Cross-file classdef constructor body analysis (property shapes now propagate)
- 462 tests

## 2.9.0

- License key system: `conformal.licenseKey` setting replaces `conformal.pro` boolean
- "Enter License Key" command in the command palette with validation
- Status bar shows "Conformal Pro" when a valid license is active
- Symbolic colon range tracking: `1:n` infers `matrix[1 x n]`
- `length()`/`numel()` propagation with symbolic dim aliasing
- 3D array slice extraction: `A(:,:,k)` from `zeros(m, n, T)` returns the 2D slice shape
- Cell variable-index read/write and 2D assignment tracking
- Multi-level nested function support (4+ levels)
- Dogfood-driven false positive fixes

## 2.8.0

- Terse diagnostic messages (GCC/Clang style)
- Warning documentation: each diagnostic code links to an explanation page
- Expanded syntax highlighting: classdef, persistent, global, parfor, 50+ builtins
- Build pipeline smoke test prevents shipping stale Fable output
- BOM handling for UTF-8 files with byte order marks
- Status bar shows "No issues" instead of "Ready" for clean files
- `conformal.inlayHints` setting: toggle shape inlay hints on first assignment (default: on)
- Sorted struct field output in hover and diagnostics for deterministic display
- Soundness fixes: `joinValueRanges`, If/IfChain break alignment, `widenDim` lattice contract
- `resolveCall` dispatch unification: all call sites route through a single discriminated union

## 2.7.0

- Inlay hints: shape annotations on first assignment (`: matrix[3 x 4]`)
- Fable/JS server at full parity with native .NET server
- Go-to-definition, hover, code actions, document symbols in both servers
- Pro-tier filtering in VS Code extension

## 2.0.0

- Complete rewrite: F# analyzer with Fable-compiled JS server
- In-process LSP via IPC transport (no Python subprocess)
- 190KB VSIX package
- 500ms debounced analyze-on-change

## 1.0.0

- Initial release with Python-based analyzer
- Shape tracking for matrix operations
- Basic LSP integration
