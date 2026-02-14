# Operation Semantic Rules

## Builtin Shape Rules (Phase 3)

### Matrix Constructors

**zeros/ones** (2-arg form):
- `zeros(r, c)` → `matrix[r x c]`
- `ones(r, c)` → `matrix[r x c]`
- Symbolic dimensions supported: `zeros(n, m)` → `matrix[n x m]`

**eye/rand/randn** (0/1/2-arg forms):
- `randn()` → `scalar` (0-arg)
- `randn(n)` → `matrix[n x n]` (1-arg square)
- `randn(m, n)` → `matrix[m x n]` (2-arg rectangular)
- Same for `eye()`, `rand()`
- Edge case: `eye(0)` → `matrix[0 x 0]`

### Element-Wise Operations

**abs, sqrt** (pass-through shape):
- `abs(x)` → `shape_of(x)`
- `sqrt(x)` → `shape_of(x)`
- Preserves scalars, matrices, and symbolic dimensions

### Transpose

**transpose() function**:
- `transpose(scalar)` → `scalar`
- `transpose(matrix[r x c])` → `matrix[c x r]` (swaps dimensions)
- Consistent with `.'` operator

**Transpose operator** (`.'`):
- `A.'` where `A = matrix[r x c]` → `matrix[c x r]`
- Handled in `eval_expr_ir()` via `Transpose` IR node
- Same swapping logic as `transpose()` function

### Query Functions

**length, numel** (return scalar):
- `length(v)` → `scalar`
- `numel(M)` → `scalar`
- Accept ranges: `length(1:10)` → `scalar`

**size** (return row vector or scalar):
- `size(A)` → `matrix[1 x 2]` (returns `[rows, cols]`)
- `size(A, dim)` → `scalar` (returns single dimension)

**isscalar** (return logical scalar):
- `isscalar(x)` → `scalar`

### Scalar-Returning Operations

**det, norm** (evaluate arg, return scalar):
- `det(A)` → `scalar` (determinant)
- `norm(A)` → `scalar` (matrix/vector norm)
- No dimension validation (MATLAB-compliant — det defined for non-square)

### Shape-Dependent Dispatch

**diag** (vector ↔ diagonal matrix):
- `diag(scalar)` → `matrix[1 x 1]`
- `diag(matrix[n x 1])` → `matrix[n x n]` (column vector → diagonal)
- `diag(matrix[1 x k])` → `matrix[k x k]` (row vector → diagonal)
- `diag(matrix[m x n])` → `matrix[None x 1]` (matrix → diagonal extraction)
- **Conservative cases**:
  - `diag(matrix[None x None])` → `matrix[None x 1]` (can't prove vectorness)
  - `diag(matrix[n x m])` (symbolic, different names) → `matrix[None x 1]` (can't prove vectorness)
- **Implementation**: Uses `== 1` check only on concrete dims; symbolic names don't spuriously match

**inv** (pass-through for square):
- `inv(matrix[n x n])` → `matrix[n x n]` (square → same shape)
- `inv(matrix[m x n])` where `m ≠ n` → `unknown` (non-square or can't prove square)
- **Edge case**: `inv(matrix[None x None])` → `matrix[None x None]` (both dims unknown, assume square)

### Row Vector Generators

**linspace** (linearly spaced points):
- `linspace(a, b)` → `matrix[1 x 100]` (MATLAB default: 100 points)
- `linspace(a, b, n)` → `matrix[1 x n]` (n points, symbolic n supported)
- Evaluates first two args for side effects

## Matrix Operations (Pre-Phase 3)

**Matrix Multiplication** (`*`):
- `matrix[n x m] * matrix[m x k]` → `matrix[n x k]`
- Inner dimension compatibility checked
- Mismatch → `W_MATMUL_INNER_DIM_MISMATCH` + `unknown`

**Concatenation** (horizontal `[A, B]` / vertical `[A; B]`):
- Horizontal: `matrix[n x m]` + `matrix[n x k]` → `matrix[n x (m+k)]`
- Vertical: `matrix[n x m]` + `matrix[k x m]` → `matrix[(n+k) x m]`
- Symbolic arithmetic: `add_dim(m, k)` → `"m+k"` if both symbolic

**Element-wise** (`+`, `-`, `.*`, `./`):
- Requires matching dimensions (or scalar broadcast)
- `matrix[n x m] + matrix[n x m]` → `matrix[n x m]`
- `matrix[n x m] + scalar` → `matrix[n x m]` (broadcast)
- Dimension mismatch → `W_ELEMENTWISE_DIM_MISMATCH` + `unknown`

## Soundness Guarantees

- All operations degrade to `unknown` when args cannot be resolved
- Unknown functions emit `W_UNKNOWN_FUNCTION` warning
- Colons in builtin args → force indexing interpretation (not call)
- ValueError exceptions caught → return `unknown`
- No false positives (over-approximation is conservative)
