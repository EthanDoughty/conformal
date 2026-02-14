# Semantic Differential Auditor Memory

## Phase 3 Builtin Shape Rules (Validated 2026-02-13)

### Verified Correct Behaviors

1. **Matrix Constructors**:
   - `eye(n)` → `matrix[n x n]` (symbolic square matrix)
   - `eye(m, n)` → `matrix[m x n]` (symbolic rectangular)
   - `eye(3)` → `matrix[3 x 3]` (concrete)
   - `eye(0)` → `matrix[0 x 0]` (edge case handled)
   - Same rules apply to `rand()`, `randn()`, `zeros()`, `ones()`
   - **1-arg square forms**: `zeros(n)` → `matrix[n x n]` ✅
   - **2-arg rectangular forms**: `zeros(m,n)` → `matrix[m x n]` ✅

2. **0-Arg Constructor Forms**:
   - `randn()` → `scalar` (MATLAB-compliant)
   - `rand()` → `scalar` (MATLAB-compliant)

3. **Element-Wise Pass-Through**:
   - `abs(matrix[2 x 3])` → `matrix[2 x 3]` ✅
   - `sqrt(scalar)` → `scalar` ✅
   - Symbolic dimensions preserved: `abs(matrix[n x m])` → `matrix[n x m]` ✅

4. **Transpose Function**:
   - `transpose(matrix[2 x 3])` → `matrix[3 x 2]` (swaps dimensions) ✅
   - Consistent with `.'` operator semantics (test5.m validates `v'`)
   - Scalars pass through: `transpose(scalar)` → `scalar` ✅
   - Symbolic swap: `transpose(matrix[n x m])` → `matrix[m x n]` ✅

5. **Query Functions**:
   - `length(vector)` → `scalar` ✅
   - `numel(matrix[n x m])` → `scalar` ✅
   - Accept ranges: `length(1:10)` → `scalar` ✅

6. **Scalar-Returning Operations**:
   - `det(matrix)` → `scalar` ✅ (evaluates arg, no dimension validation)
   - `norm(array)` → `scalar` ✅ (evaluates arg)

7. **Shape-Dependent Dispatch (diag)**:
   - `diag(scalar)` → `matrix[1 x 1]` ✅
   - `diag(matrix[n x 1])` → `matrix[n x n]` ✅ (column vector → diagonal)
   - `diag(matrix[1 x k])` → `matrix[k x k]` ✅ (row vector → diagonal)
   - `diag(matrix[m x n])` → `matrix[None x 1]` ✅ (matrix → diagonal extraction)
   - **Conservative approximation**: `diag(matrix[None x None])` → `matrix[None x 1]` (can't prove vectorness)
   - **Conservative approximation**: `diag(matrix[n x m])` (symbolic different names) → `matrix[None x 1]`
   - **Soundness**: Uses `== 1` check only on concrete dims; symbolic dims don't spuriously match

8. **Pass-Through for Square Matrices (inv)**:
   - `inv(matrix[3 x 3])` → `matrix[3 x 3]` ✅ (concrete square)
   - `inv(matrix[n x n])` → `matrix[n x n]` ✅ (symbolic square, same name)
   - `inv(matrix[3 x 4])` → `unknown` ✅ (concrete non-square)
   - `inv(matrix[n x m])` → `unknown` ✅ (symbolic different names)
   - **Edge case**: `inv(matrix[None x None])` → `matrix[None x None]` (slightly imprecise but sound)

9. **Row Vector Generators (linspace)**:
   - `linspace(a, b)` → `matrix[1 x 100]` ✅ (MATLAB default)
   - `linspace(a, b, n)` → `matrix[1 x n]` ✅ (symbolic n supported)
   - Evaluates first two args for side effects ✅

10. **Soundness (Graceful Degradation)**:
    - Unknown function calls → `unknown` shape + `W_UNKNOWN_FUNCTION` warning
    - Unparseable builtin args (e.g., colons in constructors) → fall through to indexing or `unknown`
    - ValueError exceptions caught → returns `unknown`

11. **Symbolic Dimension Tracking**:
    - Symbolic dims preserved across operations: `eye(n)` → `matrix[n x n]` ✅
    - No spurious unification detected
    - Symbolic dimensions correctly duplicated: `diag(matrix[n x 1])` → `matrix[n x n]` ✅
    - Equality checks (`r == c` in `inv`, `r == 1` in `diag`) work correctly for symbolic names ✅

### Expected IR vs Legacy Divergences

- **tests/builtins/constructors.m**: Legacy returns all `unknown` (lacks builtin rules); IR returns precise shapes ✅ EXPECTED
- **tests/builtins/unknown_function.m**: Legacy lacks `W_UNKNOWN_FUNCTION` warning; IR emits it ✅ EXPECTED
- **tests/builtins/unknown_function.m**: Legacy returns `unknown` for `randn(3,4)`; IR returns `matrix[3 x 4]` ✅ EXPECTED

### Test Coverage

- **tests/builtins/constructors.m**: Phase 1-2 builtin validation (39 shape expectations, 0 warnings)
- **tests/builtins/remaining_builtins.m**: Phase 3 builtin validation (20 shape expectations, 0 warnings)
  - Covers: det, norm, diag (all dispatch cases), inv, linspace (2/3-arg), zeros/ones 1-arg
- **tests/builtins/unknown_function.m**: Updated for `randn()` constructor + `W_UNKNOWN_FUNCTION` warning
- **tests/symbolic/dimension_tracking.m**: Validates `'` operator transpose (implicitly checks consistency with `transpose()` function)

## Links

- [operations.md](operations.md): Full operation semantic rules (builtins, matrix ops, etc.)
- [lattice-soundness.md](lattice-soundness.md): Conservative approximation patterns and soundness validation checklist
