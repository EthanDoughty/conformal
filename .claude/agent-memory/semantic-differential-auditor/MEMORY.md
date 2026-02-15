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

## Loop Widening Implementation (v0.9.2, Validated 2026-02-13)

See [widening-analysis.md](widening-analysis.md) for detailed analysis.

**Status**: ✅ SEMANTICALLY CORRECT with one theoretical concern

**Key findings**:
1. **Convergence guaranteed**: ≤2 iterations due to lattice height 2
2. **Widening operator sound**: Conflicting dims → None, stable dims preserved
3. **Post-loop join correct**: Using `widen_env` is semantically equivalent to `join_env` for this use case
4. **Unknown-as-bottom**: Intentional design for unbound variables, works correctly in practice

**Theoretical concern** (not exercised by tests):
- `widen_shape(s, unknown) → s` may be unsound if `unknown` is from explicit binding (error state)
- In practice, expressions producing `unknown` get converted to `matrix[None x ...]` by concat logic
- **Recommendation**: Add test for loop body that assigns variable to unknown

**Test validation**: All 49 tests pass with both default and `--fixpoint` modes

## v0.9.3 Bottom/Error Lattice Fix (Validated 2026-02-14)

**Status**: ✅ SEMANTICALLY CORRECT AND SOUND

### Core Change
Added `Shape.bottom()` as distinct lattice element (identity) vs `Shape.unknown()` (absorbing top/error). This fixes the soundness bug from v0.9.2 where unbound variables (bottom) and errors (top) were conflated.

### Verified Behaviors
1. **Operator tables match spec**: `widen_shape` and `join_shape` implement bottom-as-identity, unknown-as-top correctly ✅
2. **Bottom containment**: `Env.get()` returns bottom for unbound vars; Var eval converts bottom → unknown ✅
3. **Soundness fix**: `widen(matrix, unknown)` now correctly returns unknown (was matrix pre-fix) ✅
4. **Precision regression**: `join(unknown, matrix)` now returns unknown (documented in if_branch_mismatch.m) ✅
5. **All 58 tests pass** in both default and fixpoint modes ✅

### Soundness Fix Details
- **Pre-v0.9.3 bug** (tests/loops/widen_unknown_in_body.m): `widen(matrix[3x3], unknown_from_error)` incorrectly returned `matrix[3x3]`, missing error state in loop
- **Post-v0.9.3 fix**: `widen(matrix[3x3], unknown)` correctly returns `unknown`, fixpoint mode propagates error state

### Precision Regression (Sound)
- **Test**: tests/control_flow/if_branch_mismatch.m
- **Change**: `join(unknown, matrix[n x k])` → `unknown` (was `matrix[n x k]` pre-v0.9.3)
- **Rationale**: If one branch produces error (unknown), result is truly indeterminate. More sound, less precise.

### Bottom-as-Identity Use Cases
1. **First assignment in loop**: Pre-loop env.get("B") → bottom, widen(bottom, matrix[3x3]) → matrix[3x3] ✅
2. **Variable in one branch only**: join(matrix[3x3], bottom) → matrix[3x3] ✅

## v0.10.0 User-Defined Functions (Validated 2026-02-14)

**Status**: ✅ SEMANTICALLY CORRECT AND SOUND

### Core Features Validated

1. **Lattice soundness** (lines 196-200 in analysis_ir.py):
   - Bottom→unknown conversion for unset output vars: `func_env.get(out_var).is_bottom()` → `Shape.unknown()` ✅
   - Maintains v0.9.3 lattice invariants: bottom-as-identity, unknown-as-top ✅
   - Verified via operator tables: `widen_shape`, `join_shape` match specification ✅

2. **Dimension aliasing correctness** (lines 141-145 in analysis_ir.py):
   - `expr_to_dim_ir(Var("param"), func_env)` checks `env.dim_aliases` first ✅
   - Aliasing propagates caller dimension names: `make_matrix(n, m)` → `matrix[n x m]` (not `matrix[rows x cols]`) ✅
   - Symbolic arithmetic with aliases: `zeros(rows+1, cols)` → `matrix[(n+1) x m]` when aliased ✅
   - Env.copy() preserves dim_aliases: fresh workspace isolation maintained ✅

3. **Recursion guard soundness** (lines 124-129, 205-207 in analysis_ir.py):
   - Direct recursion detected: `if func_name in ctx.analyzing_functions` ✅
   - Returns `[Shape.unknown()] * max(len(sig.output_vars), 1)` ✅
   - Mutual recursion detected: both functions added to set before body analysis ✅
   - **Cleanup guarantee**: `finally: ctx.analyzing_functions.discard(func_name)` ✅
   - Prevents false recursion on sequential calls to same function ✅

4. **Warning propagation** (lines 152-190 in analysis_ir.py):
   - Dual-location format: `"Line 8 (in func_name, called from line 12): ..."` ✅
   - Parses both `"Line N:"` and `"W_CODE line N:"` formats ✅
   - Skips rewrite if warning already has `"(in "` context (nested calls) ✅
   - Tested with: dimension mismatch, unknown function, recursive call warnings ✅

5. **AssignMulti correctness** (lines 312-348 in analysis_ir.py):
   - Correct count: `[A, B] = two_returns(...)` binds both targets ✅
   - Count mismatch: emits `W_MULTI_ASSIGN_COUNT_MISMATCH`, sets all targets to unknown ✅
   - Non-call RHS: emits `W_MULTI_ASSIGN_NON_CALL`, sets all targets to unknown ✅
   - Builtin on RHS: emits `W_MULTI_ASSIGN_BUILTIN`, sets all targets to unknown ✅
   - Destructuring validated with `tests/functions/multiple_returns.m` ✅

6. **Two-pass analysis** (lines 67-80 in analysis_ir.py):
   - Pass 1: Registers all FunctionDef nodes in `ctx.function_registry` ✅
   - Pass 2: Analyzes script statements (skips FunctionDef) ✅
   - Function names don't pollute script environment ✅
   - Functions can call other functions defined later in file ✅

7. **Fresh workspace isolation** (line 133 in analysis_ir.py):
   - Each call creates `func_env = Env()` (empty bindings) ✅
   - Caller's variables not visible in function body ✅
   - Parameter binding doesn't leak to caller ✅
   - Verified: `func_env.get("caller_var")` → `bottom` ✅

### Test Coverage

- **tests/functions/simple_function.m**: Single-arg, single-return (0 warnings)
- **tests/functions/multiple_returns.m**: Destructuring `[A, B] = func(...)` (0 warnings)
- **tests/functions/matrix_constructor.m**: Dimension aliasing `zeros(rows, cols)` → `matrix[n x m]` (0 warnings)
- **tests/functions/procedure.m**: Procedure detection (no return value) → `W_PROCEDURE_IN_EXPR`
- **tests/functions/unknown_in_function.m**: Warning propagation with dual-location context
- **tests/functions/function_then_script.m**: Two-pass analysis (0 warnings)
- **tests/functions/call_with_mismatch.m**: Dual-location for dimension mismatch in body
- **tests/functions/recursion.m**: Recursion guard → `W_RECURSIVE_FUNCTION`

**All 66 tests pass** (58 pre-existing + 8 new function tests)

### Edge Cases Validated

1. **Unset output variable**: Returns `unknown` (bottom→unknown conversion) ✅
2. **Multiple sequential calls**: Recursion guard cleanup allows non-recursive re-calls ✅
3. **Mutual recursion**: First call adds to set, second call detects recursion ✅
4. **Nested function calls**: Warning context already has `"(in ..."`, no double-rewrite ✅
5. **Dimension alias isolation**: Each function call gets fresh `dim_aliases` dict via `Env()` ✅
6. **Env.copy() preserves aliases**: Control flow joins (if/else in function body) maintain aliasing ✅

### Implementation Quality

- **Exception safety**: `finally` block guarantees recursion guard cleanup ✅
- **Deduplication**: `warnings = list(dict.fromkeys(warnings))` prevents duplicate warnings ✅
- **Defensive coding**: Max count for unknown returns: `max(len(sig.output_vars), 1)` ✅
- **Lattice correctness**: All shape operations respect bottom-as-identity, unknown-as-top ✅

## Links

- [operations.md](operations.md): Full operation semantic rules (builtins, matrix ops, etc.)
- [widening-analysis.md](widening-analysis.md): Loop widening detailed analysis (v0.9.2)
- [lattice-soundness.md](lattice-soundness.md): Bottom/error lattice fix detailed analysis (v0.9.3) + conservative approximation patterns
