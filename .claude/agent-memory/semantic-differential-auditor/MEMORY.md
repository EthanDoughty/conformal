# Semantic Differential Auditor Memory

## Phase 3 Builtin Shape Rules (Validated 2026-02-13)

### Verified Correct Behaviors

1. **Matrix Constructors**:
   - `eye(n)` → `matrix[n x n]` (symbolic square matrix)
   - `eye(m, n)` → `matrix[m x n]` (symbolic rectangular)
   - `eye(3)` → `matrix[3 x 3]` (concrete)
   - `eye(0)` → `matrix[0 x 0]` (edge case handled)
   - Same rules apply to `rand()`, `randn()`

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

6. **Soundness (Graceful Degradation)**:
   - Unknown function calls → `unknown` shape + `W_UNKNOWN_FUNCTION` warning
   - Unparseable builtin args (e.g., colons in constructors) → fall through to indexing or `unknown`
   - ValueError exceptions caught → returns `unknown`

7. **Symbolic Dimension Tracking**:
   - Symbolic dims preserved across operations: `eye(n)` → `matrix[n x n]` ✅
   - No spurious unification detected
   - Symbolic arithmetic not tested in Phase 3 (concat not involved)

### Expected IR vs Legacy Divergences

- **tests/builtins/constructors.m**: Legacy returns all `unknown` (lacks builtin rules); IR returns precise shapes ✅ EXPECTED
- **tests/builtins/unknown_function.m**: Legacy lacks `W_UNKNOWN_FUNCTION` warning; IR emits it ✅ EXPECTED
- **tests/builtins/unknown_function.m**: Legacy returns `unknown` for `randn(3,4)`; IR returns `matrix[3 x 4]` ✅ EXPECTED

### Test Coverage

- **tests/builtins/constructors.m**: Comprehensive Phase 3 builtin validation (39 shape expectations, 0 warnings)
- **tests/builtins/unknown_function.m**: Updated for `randn()` constructor + `W_UNKNOWN_FUNCTION` warning
- **tests/symbolic/dimension_tracking.m**: Validates `'` operator transpose (implicitly checks consistency with `transpose()` function)

## Links

See [operations.md](operations.md) for full operation semantic rules.
