# Conformal Analysis Details

This document covers the full technical details of Conformal's analysis capabilities. For a summary, see the [README](../README.md). For the full test listing, see [tests.md](tests.md). For individual warning code documentation, see the [warnings/](warnings/) directory.

## What the Analysis Detects

All warnings include source line numbers and are formatted in a terse GCC/Clang style, so they can be parsed by editors and CI tools. When Conformal finds a definite error, it marks the result as unknown and keeps going so as many diagnostics as possible are reported in a single pass. The parser also accumulates recovered parse errors with token-level span information, so multiple syntax issues in the same file can be reported at once rather than stopping at the first one.

### Operations

Conformal detects dimension mismatches in `+`, `-`, `*`, `.*`, `./`, `^`, `.^`. Scalar-matrix broadcasting (e.g. `s*A`, `s + A`) is handled. Backslash `A\b` (mldivide) follows the same inner-dimension logic that multiplication does. Element-wise logical `&` and `|` pass shapes through like an element-wise op usually would. Logical NOT `~` and dot-transpose `.'` also carry shapes through. If `*` is used where `.*` was probably intended, Conformal suggests the fix.

Comparison operators (`==`, `~=`, `<`, `<=`, `>`, `>=`) return the broadcast shape of their operands, so `A > 0` where `A` is `matrix[3 x 4]` gives `matrix[3 x 4]` rather than scalar. This means logical indexing `A(A > 0)` is recognized as a matrix-typed operation, and Conformal infers the result as `matrix[None x 1]` (a column vector, since logical indexing in MATLAB always returns a column) rather than treating it as an unknown scalar index. Row vector inputs preserve orientation, giving `matrix[1 x None]` instead.

### Literals and concatenation

Conformal handles matrix literals like `[1 2; 3 4]`, cell array literals like `{1, 2; 3, 4}`, and string literals in both quote styles (`'hello'`, `"world"`). Horizontal concatenation `[A B]` checks that row counts match, and vertical concatenation `[A; B]` checks columns. Symbolic dimensions compose through concatenation, so `[A B]` where A is `n x k` and B is `n x m` gives `n x (k+m)`. An empty matrix `[]` is treated as the identity element for concatenation, so `[[] x]` and `[[] ; x]` both simplify to `x` with no false mismatch warning. Matrix literal spacing is handled correctly too, so `[1 -2; 3 -4]` parses as four elements rather than as subtraction.

### Indexing

Conformal supports parenthesized indexing `A(i,j)`, slice indexing `A(:,j)` and `A(i,:)`, range indexing `A(2:5,:)`, linear indexing, and full-matrix `A(:,:)`. Curly-brace indexing `C{i,j}` works for cell arrays with per-element shape tracking, where a literal index extracts the precise element shape. The `end` keyword works in indexing contexts with arithmetic support (`C{end}`, `C{end-1}`, `A(1:end, 2)`, `A(end-2:end, :)`). Indexed assignment `M(i,j) = expr` preserves the matrix shape, and the write side doesn't check bounds, since MATLAB auto-expands arrays on assignment. Read-side out-of-bounds checking is unchanged.

### Functions

Over 635 MATLAB builtins are recognized (so calls to them don't produce `W_UNKNOWN_FUNCTION` warnings), and around 315 of those have explicit shape rules. First, matrix constructors like `zeros`, `ones`, `eye`, `rand`, `randn`, `true`, `false`, `nan`, and `inf` handle all three call forms: no args gives a scalar, one arg gives an n x n square, and two args give an m x n matrix. Second, shape transformations include `reshape` (with a conformability check), `repmat`, `diag`, `transpose`, `horzcat`, `vertcat`, and more specialized ones like `kron` (where `kron(A[m x n], B[p x q])` gives `matrix[(m*p) x (n*q)]`) and `blkdiag` (where `blkdiag(A[m x n], B[p x q])` gives `matrix[(m+p) x (n+q)]`). Third, element-wise math functions like `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `ceil`, `floor`, and their relatives all pass the input shape through. Reductions like `sum`, `prod`, `mean`, `min`, `max`, and `diff` accept an optional dimension argument. Finally, type predicates like `isscalar`, `iscell`, `isempty`, and `isnumeric` return scalar, and linear algebra functions like `det`, `inv`, `norm`, and `linspace`, query functions like `size`, `length`, and `numel`, and two-argument element-wise functions like `mod`, `rem`, and `atan2` are also covered.

Dimension arithmetic works inside builtin arguments, so `zeros(n+1, 2*m)` is tracked symbolically.

Higher-order builtins `cellfun` and `arrayfun` are also supported. For calls like `cellfun(@func, C)` or `cellfun(@(x) expr, C)`, Conformal dispatches to the provided handle or lambda to figure out the per-element output shape, then combines that with the cell or matrix dimensions to infer the result. Passing `'UniformOutput', false` returns a cell array matching the input dimensions.

Conformal analyzes user-defined functions at each call site with the caller's argument shapes. Three forms are supported: single return (`function y = f(x)`), multi-return (`function [a, b] = f(x)`), and procedures (including no-arg `function name` syntax). Nested `function...end` blocks inside another function body are also supported, with read/write access to the parent workspace via scope chains and forward-reference visibility between siblings. The parser also handles pre-2016 end-less function definitions (files where `end` is omitted), and space-separated multi-return syntax (`function [a b c] = f(...)` without commas). Anonymous functions `@(x) expr` are analyzed the same way, with by-value closure capture at definition time. Function handles `@funcName` dispatch to their targets. Results are cached per argument shape tuple so the same function called with the same shapes isn't re-analyzed. The cache key includes the argument count, so a function called with different numbers of arguments is analyzed separately.

Optional argument patterns using `nargin` and `nargout` are supported. When a function is called with fewer arguments than it declares, the call isn't flagged as an error, since the missing arguments could be optional. Inside the function body, `nargin` is bound to the exact argument count as a concrete interval `[n, n]`, so `if nargin < 3` can be refined precisely and the default-value branch is analyzed correctly. `nargout` works the same way: it carries the number of requested outputs, so `if nargout > 1` can prune dead branches. Calling with more arguments than the function declares is still an error.

Functions using `varargin` as the last parameter are also supported. Extra call arguments beyond the named ones are bundled into a cell with per-element shape tracking, so `varargin{1}` returns the actual shape of the first extra argument, not just unknown. The arg-count warning is suppressed when `varargin` is present. `varargout` works at the call site: output targets beyond the named return variables receive unknown shape.

When analyzing a file, Conformal also scans sibling `.m` files in the same directory and fully analyzes their bodies (parse -> analyze) to infer real return shapes. Dimension aliasing works across file boundaries, subfunctions in external files are supported, and cross-file cycles (A->B->A) are detected and handled gracefully. Unparseable external files emit `W_EXTERNAL_PARSE_ERROR`. Classdef files are also scanned cross-file: if a sibling `.m` file is a `classdef`, Conformal extracts its property list and method signatures lazily on first use, so `MyClass(args)` and `obj.method(args)` work even when the class definition lives in a separate file.

### Data structures

Conformal tracks struct field assignment (`s.x = A`), field access, and chained dot notation (`s.x.y`). Missing field access on a known struct emits a warning. Struct shapes join across branches by taking the union of fields. Assigning a field to a variable whose base shape is unknown, for example the return value of an unrecognized function, creates an open struct, written `struct{x: matrix[1 x 3], ...}`. Open structs don't warn on missing field access, since there could be more fields that aren't tracked. Multi-return destructuring supports dotted targets, so `[s.x, s.y] = get_pair()` works and populates the struct's field map. The lattice ordering is bottom < closed struct < open struct < unknown.

Cell arrays work with `cell(n)` and `cell(m,n)` constructors, curly-brace indexing, and element assignment. Literal indexing `C{i}` extracts the precise shape of element `i` when available, and dynamic indexing joins all element shapes conservatively. Curly-brace indexing on a non-cell emits a warning.

Basic `classdef` support is included through a side-channel approach. When the parser encounters a `classdef` block, it extracts the property names and the constructor method body. A class registry maps the class name to its property list and constructor definition, so calling the constructor analyzes the body and returns a struct with the declared fields. Fields like `obj.x` are tracked the same way regular struct fields are. Calling `obj.method(args)` checks whether `obj` was created by a known class constructor, and if so dispatches the call as `method(obj, args)` through the class's registered method definitions. Superclass syntax (`classdef Foo < Bar`) parses correctly, and the superclass name is stored, but inheritance is not resolved.

### Control flow

Conformal covers `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`/`otherwise`, `try`/`catch`, `break`, `continue`, and `return`. When branches assign different shapes to the same variable, Conformal joins them conservatively. Loops use a single pass by default, or widening-based fixed-point iteration via `--fixpoint` for guaranteed convergence in at most 2 iterations. In `--fixpoint` mode, for-loop accumulation patterns like `A = [A; delta]` and `A = [A, delta]` are detected and refined algebraically: the iteration count is extracted from the range (`(b-a)+1`), and the widened dimension is replaced with `init_dim + iter_count * delta_dim`. Interval widening in fixpoint loops produces sound post-loop intervals, so interval-based checks remain accurate even after a fixpoint run.

Switch/case bodies also benefit from interval refinement: when the switch expression is an integer variable and a case arm matches a concrete value, Conformal narrows the variable to that value inside the case body, which can eliminate false-positive dimension and bounds warnings in dispatch-style code. When a case arm uses a cell list like `case {1, 5}`, Conformal computes the hull interval of all the listed integers and narrows the switch variable to that hull, so `zeros(n, n)` inside such an arm can resolve to a bounded size.

### Symbolic dimensions

Variables with unknown concrete size get symbolic names like `n`, `m`, `k`, and those names propagate through operations: `1:n` gives a `1 x n` vector, `[A B]` computes `n x (k+m)`, `zeros(n+1, 2*m)` tracks the arithmetic. Symbolic dimensions are represented as canonical polynomials with rational coefficients, so `n+m` and `m+n` are recognized as equal, and `n+n` simplifies to `2*n`. When a function is called with a symbolic argument `n`, the dimension name propagates into the function body.

### Interval analysis

In parallel with shape inference, Conformal tracks scalar integer variables through an integer interval domain `[lo, hi]`. This is what enables three additional checks: `W_DIVISION_BY_ZERO` when the divisor is provably zero, `W_INDEX_OUT_OF_BOUNDS` when an index is provably outside the matrix dimensions, and `W_POSSIBLY_NEGATIVE_DIM` when a dimension expression is provably non-positive.

Initially, for-loop variables are bound to their range interval, so `for i = 1:n` records `i` in `[1, n]` with a symbolic upper bound. Comparisons against symbolic bounds fall back soundly, and intervals join conservatively across control-flow branches.

When `--fixpoint` widening would ordinarily snap an interval bound to infinity, Conformal instead snaps to the nearest threshold in the set `{-1000, -100, -10, -1, 0, 1, 10, 100, 1000}`. A counter that grows inside a loop widens its upper bound to `1000` rather than `+inf`, so the interval stays finite and downstream index checks remain useful. This is a standard technique from abstract interpretation called threshold widening.

Additionally, branch conditions narrow variable intervals inside the branch body. A condition like `if x > 0` causes Conformal to refine `x` to `[1, +inf]` for the true branch, which can eliminate false-positive out-of-bounds and negative-dim warnings when a guard proves safety. Conformal supports `>`, `>=`, `<`, `<=`, `==`, `~=`, compound `&&` conditions, and operator flipping like `5 >= x`.

There is also a cross-domain bridge between the interval domain and the dimension equivalence classes. When an exact interval `[k, k]` is recorded for a variable, for example because a branch condition like `if r == 5` narrows `r` to a singleton, the bridge propagates `k` into any DimEquiv equivalence class that `r` belongs to, and back into `valueRanges` for all equivalent variables. This means that if `r = size(A, 1)` and `A` has a symbolic `n` row dimension, narrowing `r` inside a branch immediately resolves `n` to that same concrete value for the duration of the branch. Similarly, `n = size(A, 1)` where `A` has a concrete row dimension now directly sets `valueRanges[n] = [dim, dim]`, so a subsequent `zeros(n, n)` can resolve to a concrete shape rather than staying symbolic.

Three more precision features operate in `--fixpoint` mode. Narrowing after widening adds a Phase 2.5 pass that re-runs the loop body once after the fixpoint stabilizes, then intersects the result with the widened bounds to tighten them back, so a counter that widened to `[0, 1000]` might recover a tighter bound, all without risking non-termination. Scope-limited widening restricts widening and narrowing operations to the variables that are actually assigned inside the loop body, so a variable like `n = 5` that is read but never written inside the loop keeps its exact `[5, 5]` interval and won't be corrupted by bridge side-effects from other variables. The Pentagon domain tracks relational upper bounds of the form `x <= y + c` and lower bounds of the form `x >= y + c`: for-loop variables get an entry `i <= n` when the range endpoint is a named variable, and a matching `i >= start` entry when the start is a named variable; when those bound variables have exact intervals, the Pentagon bridge fires to tighten `i`'s bounds from above and below. The Pentagon domain also actively suppresses `W_INDEX_OUT_OF_BOUNDS` when it can prove an index stays in bounds: `pentagonProvesInBounds` checks both the concrete case (the index interval's upper bound matches the dimension exactly) and the symbolic case (the Pentagon bound variable is in the same DimEquiv equivalence class as the matrix dimension), and `pentagonProvesLowerBound` handles the symmetric lower check to suppress "index may be < 1" warnings, so `for i = start:n; A(i,1); end` where `start >= 1` produces no spurious out-of-bounds warning. The Pentagon bridge fires before each analysis pass in `analyzeLoopBody` rather than only at loop entry, so it stays effective across all fixpoint phases. While-loop conditions are also parsed for relational bounds via `extractPentagonBoundsFromCondition`, which handles `<=`, `<`, `>=`, `>`, and `&&` conjunctions, so `while i <= n` records `i <= n` in the Pentagon domain and the bridge suppresses OOB warnings inside the loop body the same way a for-loop would.

### Type errors

Using a non-numeric type (struct, cell, function_handle) where a numeric value is expected causes Conformal to emit a type mismatch error. Arithmetic operations like `+`, `-`, `*`, and `.*` on structs or cells emit `W_ARITHMETIC_TYPE_MISMATCH`. Transpose on a non-numeric type emits `W_TRANSPOSE_TYPE_MISMATCH`, negation emits `W_NEGATE_TYPE_MISMATCH`, and mixing incompatible types in a matrix literal (like `[s, A]` where `s` is a struct) emits `W_CONCAT_TYPE_MISMATCH`. All four codes are Error severity, not warnings.

### Witness generation

For dimension conflict warnings, Conformal can optionally produce a concrete counterexample proving the bug is real. A witness is a set of variable assignments like `n=3, m=5` under which the conflicting dimensions evaluate to different integers. The solver pulls from variables with known concrete values, interval bounds from interval analysis, and branch path conditions to narrow the candidate space before enumerating. It bails out conservatively when dimensions are unknown, when symbolic terms are quadratic or higher, or when more than 8 free variables are involved.

In `--witness enrich` mode (the default when `--witness` is given), the concrete assignment is printed below each warning. In `--witness filter` mode, only warnings with a verified witness are shown, which gives you zero false positives. In `--witness tag` mode, each warning is prefixed with `[confirmed]` or `[possible]`. The LSP server always runs witness generation and appends the witness explanation to the hover text automatically, with no extra configuration needed.

### MATLAB Coder compatibility

The `--coder` flag adds a post-analysis pass that checks for constructs MATLAB Coder can't handle. Six new warning codes are emitted, all strict-only by default so they don't appear in a normal CI run unless opted in. `W_CODER_VARIABLE_SIZE` fires when a variable has an unknown or `None` dimension, since Coder requires all sizes to be statically determined. `W_CODER_CELL_ARRAY` fires when any cell array variable is present, since Coder does not support cells by default. `W_CODER_DYNAMIC_FIELD` fires on dynamic struct field access `s.(expr)`. `W_CODER_TRY_CATCH` fires on try/catch blocks, which Coder restricts. `W_CODER_UNSUPPORTED_BUILTIN` fires on calls to one of the 30 builtins that Coder does not support, including `eval`, `feval`, `input`, `keyboard`, and display functions like `disp` and `fprintf`. `W_CODER_RECURSION` fires on recursive function calls.

Passing `--coder --strict` together surfaces all six codes, or combine with `--fixpoint` for the most precise size analysis before the Coder pass runs.

## Warning Code Catalog

Conformal has 53 warning codes organized into three tiers. Each code has its own documentation page in the [warnings/](warnings/) directory. The tier determines when the warning is shown:

### Default tier (36 codes, always shown)

These fire in default mode with no configuration required.

| Code | Category | What it catches |
|------|----------|----------------|
| `W_INNER_DIM_MISMATCH` | Shape | Inner dimensions don't match in matrix multiply |
| `W_ELEMENTWISE_MISMATCH` | Shape | Operand shapes differ in element-wise operation |
| `W_HORZCAT_ROW_MISMATCH` | Shape | Row counts differ in horizontal concatenation |
| `W_VERTCAT_COL_MISMATCH` | Shape | Column counts differ in vertical concatenation |
| `W_RESHAPE_MISMATCH` | Shape | Element count changes in reshape |
| `W_MLDIVIDE_DIM_MISMATCH` | Shape | Dimensions incompatible for backslash solve |
| `W_MATRIX_POWER_NON_SQUARE` | Shape | Matrix power on a non-square matrix |
| `W_CONCAT_TYPE_MISMATCH` | Type | Incompatible types in a matrix literal |
| `W_ARITHMETIC_TYPE_MISMATCH` | Type | Arithmetic on a non-numeric type (struct, cell, handle) |
| `W_TRANSPOSE_TYPE_MISMATCH` | Type | Transpose on a non-numeric type |
| `W_NEGATE_TYPE_MISMATCH` | Type | Negation on a non-numeric type |
| `W_NOT_TYPE_MISMATCH` | Type | Logical NOT on a non-numeric type |
| `W_RANGE_NON_SCALAR` | Indexing | Range endpoint is not scalar |
| `W_INVALID_RANGE` | Indexing | Range is provably empty or malformed |
| `W_NON_SCALAR_INDEX` | Indexing | Non-scalar used as a single index |
| `W_INDEX_ASSIGN_TYPE_MISMATCH` | Indexing | Indexed assignment into a non-indexable type |
| `W_INDEX_OUT_OF_BOUNDS` | Intervals | Index provably outside matrix dimensions |
| `W_DIVISION_BY_ZERO` | Intervals | Divisor provably zero |
| `W_POSSIBLY_NEGATIVE_DIM` | Intervals | Dimension expression provably non-positive |
| `W_CONSTRAINT_CONFLICT` | Constraints | Symbolic dimension constraint is unsatisfiable |
| `W_UNKNOWN_FUNCTION` | Cross-file | Unrecognized function call (not a builtin, not in workspace) |
| `W_EXTERNAL_PARSE_ERROR` | Cross-file | Sibling `.m` file could not be parsed |
| `W_RECURSIVE_FUNCTION` | Cross-file | Recursive function call detected |
| `W_STRUCT_FIELD_NOT_FOUND` | Type tracking | Field access on a closed struct with no such field |
| `W_FIELD_ACCESS_NON_STRUCT` | Type tracking | Dot access on a non-struct value |
| `W_CURLY_INDEXING_NON_CELL` | Type tracking | Curly-brace indexing on a non-cell value |
| `W_CELL_ASSIGN_NON_CELL` | Type tracking | Cell element assignment on a non-cell value |
| `W_FUNCTION_ARG_COUNT_MISMATCH` | Functions | Too many arguments passed to a function |
| `W_PROCEDURE_IN_EXPR` | Functions | Procedure (no return value) used in an expression |
| `W_MULTI_ASSIGN_COUNT_MISMATCH` | Functions | Output count doesn't match function returns |
| `W_LAMBDA_ARG_COUNT_MISMATCH` | Functions | Lambda called with wrong number of arguments |
| `W_RETURN_OUTSIDE_FUNCTION` | Control flow | `return` at script level |
| `W_BREAK_OUTSIDE_LOOP` | Control flow | `break` outside a loop |
| `W_CONTINUE_OUTSIDE_LOOP` | Control flow | `continue` outside a loop |
| `W_END_OUTSIDE_INDEXING` | Parser | `end` keyword used outside an indexing context |
| `W_MATRIX_LIT_EMPTY_ROW` | Parser | Empty row in a matrix literal |

### Strict tier (11 codes, shown with `--strict`)

These are informational or lower-confidence codes that can be noisy. They are suppressed by default so default mode works well in CI without false-positive noise, and strict mode gives a fuller picture when needed.

| Code | Category | What it catches |
|------|----------|----------------|
| `W_UNSUPPORTED_STMT` | Parser | Statement syntax that Conformal can't fully analyze |
| `W_REASSIGN_INCOMPATIBLE` | Shape | Variable reassigned to an incompatible shape |
| `W_RECURSIVE_LAMBDA` | Functions | Self-referencing lambda detected |
| `W_LAMBDA_CALL_APPROXIMATE` | Functions | Lambda call could not be precisely resolved |
| `W_MULTI_ASSIGN_NON_CALL` | Functions | Multi-return destructuring on a non-call expression |
| `W_SUSPICIOUS_COMPARISON` | Comparison | Matrix-scalar comparison that is likely a bug |
| `W_MATRIX_COMPARISON` | Comparison | Matrix-matrix comparison producing a matrix of booleans |
| `W_LOGICAL_OP_NON_SCALAR` | Comparison | Logical `&&` or `\|\|` on a non-scalar operand |
| `W_STRING_ARITHMETIC` | Type | Arithmetic involving a string value |
| `W_TOO_MANY_INDICES` | Indexing | More index dimensions than the matrix has (2D limit) |
| `W_CELLFUN_NON_UNIFORM` | Functions | `cellfun` lambda returns non-scalar without `'UniformOutput', false` |

### Coder tier (6 codes, shown with `--coder`)

These check for MATLAB Coder compatibility and only fire when the `--coder` flag is given.

| Code | What it catches |
|------|----------------|
| `W_CODER_VARIABLE_SIZE` | Variable has unknown or `None` dimension |
| `W_CODER_CELL_ARRAY` | Cell array present (Coder doesn't support cells by default) |
| `W_CODER_DYNAMIC_FIELD` | Dynamic struct field access `s.(expr)` |
| `W_CODER_TRY_CATCH` | try/catch block (Coder restricts these) |
| `W_CODER_UNSUPPORTED_BUILTIN` | One of 30 builtins Coder can't handle |
| `W_CODER_RECURSION` | Recursive function call |

### Diagnostic output format

Conformal formats diagnostics in a terse style similar to GCC and Clang, with the warning code, source location, and a one-line message. When Conformal can't determine a shape, it reports "Shape assumed unknown" rather than silently dropping the variable from tracking. This conservative approach means you always know when the analyzer has reached the limits of its precision.

### Parser error recovery

The parser uses error recovery to report multiple syntax issues in a single pass. When it encounters an unparseable construct, it records the error with token-level span information (start line, start column, end line, end column) and continues parsing. This means a file with three syntax issues will report all three rather than stopping at the first one. Recovered parse errors are accumulated alongside the normal parse tree and surfaced as `W_UNSUPPORTED_STMT` diagnostics in the output. The token span information is used by the VS Code extension to render precise underlines under the problematic syntax.

Command-style function calls (like `hold on`, `axis equal`, `grid minor`, `close all`) are recognized by the parser and suppressed from `W_UNSUPPORTED_STMT` output, since they are valid MATLAB that Conformal can't fully analyze but can safely ignore. Over 50 common command-style builtins are in the suppression list, covering plotting, display, debugging, and workspace management functions.

## Dimension Equivalence Classes

Conformal maintains a union-find data structure (DimEquiv) that tracks which symbolic dimensions are known to be equal. When a matrix multiply `A * B` succeeds, the inner dimensions `A.cols` and `B.rows` are recorded as equivalent. When `n = size(A, 1)` is called, the resulting variable `n` is linked to the row dimension of `A`.

This equivalence information propagates transitively: if `A.cols == B.rows` and `B.rows == C.rows`, then `A.cols == C.rows` is also known. When any member of an equivalence class is resolved to a concrete value (for example, through a branch condition like `if n == 5`), the resolution propagates to all members of the class via the cross-domain bridge.

The DimEquiv store is scoped to functions, so equivalences established inside a function body don't leak back to the caller. This prevents internal implementation details from polluting the caller's analysis context.

## Language Coverage

Conformal parses and tracks shapes through:

| Category | Constructs |
|----------|-----------|
| Expressions | `+`, `-`, `*`, `.*`, `./`, `^`, `.^`, `\`, `&`, `\|`, `==`, `~=`, `<`, `>`, `<=`, `>=`, `&&`, `\|\|`, `~`, `'`, `.'` |
| Literals | `[1 2; 3 4]`, `{1, 2; 3, 4}`, `'string'`, `"string"`, `1:n` |
| Indexing | `A(i,j)`, `A(:,j)`, `A(2:5,:)`, `C{i}`, `C{i} = x` |
| Assignment | `x = expr`, `s.field = expr`, `C{i} = expr`, `M(i,j) = expr`, `[a, b] = f(x)`, `[~, b] = f(x)`, `[s.x, s.y] = f(x)` |
| Functions | `function y = f(x)`, `function name` (no-arg), nested `function` blocks, `@(x) expr`, `@funcName`, `nargin`/`nargout`, `varargin`/`varargout`, `global`/`persistent` variables, `cellfun`/`arrayfun` dispatch, 635 recognized builtins (315 with shape rules) |
| Control flow | `if`/`elseif`/`else`, `for`, `while`, `switch`/`case`, `try`/`catch` |
| Statements | `break`, `continue`, `return` |
| Data types | scalars, matrices, strings, structs, cell arrays, function handles, classdef objects (constructor + method dispatch) |

## Shape System

Every expression Conformal analyzes gets a shape from an abstract domain. There are 7 user-visible shapes (plus an internal `bottom` for unbound variables):

| Shape | Example | Notes |
|-------|---------|-------|
| `scalar` | `5`, `x` | Single numeric value |
| `matrix[r x c]` | `matrix[3 x 4]`, `matrix[n x m]` | Dimensions can be concrete, symbolic, or unknown |
| `string` | `'hello'` | Char array |
| `struct{...}` | `struct{x: scalar, ...}` (open), `struct{x: scalar}` (closed) | Tracks fields and their shapes; open structs (from unknown bases) suppress missing-field warnings |
| `function_handle` | `@(x) x'`, `@sin` | Tracks lambda ID for join precision |
| `cell[r x c]` | `cell[3 x 1]` | Cell array with optional per-element shape tracking |
| `unknown` | | Error or indeterminate; the lattice top |

The shape domain is implemented as a discriminated union in F# (`Shapes.fs`), with `bottom` serving as the lattice identity, used internally for unbound variables and never surfaced in output.

Dimensions in `matrix[r x c]` can be concrete integers like `3` or `100`, symbolic names like `n`, `m`, `k`, symbolic expressions like `n+m`, `2*k`, or `n+1`, or simply unknown when no information is available.

Symbolic dimensions use a polynomial representation with rational coefficients. Canonicalization handles commutativity (`n+m` = `m+n`), like-term collection (`n+n` = `2*n`), and constant-offset conflict detection. When control flow branches assign conflicting dimensions to the same variable, Conformal joins them to unknown. In loops with `--fixpoint`, conflicting dimensions get widened to unknown while stable dimensions are preserved.

## Lattice Validation

In addition to the 515 `.m` integration tests, the shape domain itself is validated by 28 property-based tests using FsCheck. These cover 6 sections of the lattice:

1. Join commutativity: `join(a, b) == join(b, a)` for all shape pairs
2. Join associativity: `join(join(a, b), c) == join(a, join(b, c))`
3. Join monotonicity: `a <= join(a, b)` for all shapes
4. Lattice ordering consistency: `a <= b` implies `join(a, b) == b`
5. Interval join commutativity and associativity
6. Interval ordering consistency

These tests generate random shapes and intervals and verify the algebraic properties hold, catching edge cases that handwritten tests can miss. They run as part of `dotnet run -- --tests`.

## Limitations

Conformal analyzes a subset of MATLAB. Here's what it doesn't cover:

| Category | What's missing |
|----------|---------------|
| Scope | Workspace analysis covers sibling `.m` files, `addpath` directories (with `fullfile` and `genpath` constant folding), and recursive depth scanning. Cross-directory function resolution is supported. |
| Functions | `varargin` and `varargout` are supported. No `eval`. `str2func` with string literal arguments is resolved. `nargin`/`nargout` are tracked with concrete intervals at each call site. Nested functions are supported (read/write parent scope, sibling calls, forward references). `global` and `persistent` variables are tracked. |
| Builtins | 646+ builtins recognized (including Control System, Signal Processing, Aerospace, Optimization, Mapping, Image Processing, Robotics, Statistics, Communications, Computer Vision, Deep Learning, and Symbolic Math Toolbox functions); 325+ have explicit shape rules. Unrecognized calls produce a `W_UNKNOWN_FUNCTION` warning. |
| Cell arrays | Per-element tracking available for literal-indexed cells. Dynamic indexing conservatively joins all elements. |
| Indexing | `end` keyword supported with arithmetic (`C{end}`, `C{end-1}`, `A(1:end)`, `A(end-2:end, :)`). Variable operands in `end` arithmetic fall through to conservative join. |
| Data types | `classdef` support: properties extracted, constructor analyzed, `obj.method(args)` dispatched through registered method definitions, superclass inheritance chain walking for properties and methods. No maps, no tables, no N-D arrays (only 2-D matrices). No complex number tracking. |
| Syntax | No command-style calls (`save file.mat`). `global`/`persistent` variables are tracked across function boundaries. `parfor` is treated as a regular `for` loop. |
| I/O and graphics | `load`, `save`, `fprintf`, `plot`, and other side-effecting functions are recognized (no spurious `W_UNKNOWN_FUNCTION`) but their return shapes are not tracked. |
| Dynamic features | No `eval`, no `str2func`. Dynamic field access `s.(expr)` is parsed and evaluates to `unknown`. Runtime type introspection beyond type predicates (`iscell`, `isscalar`, `isnumeric`, etc.) is not tracked. |
