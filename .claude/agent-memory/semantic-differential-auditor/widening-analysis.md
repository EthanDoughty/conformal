# Loop Widening Analysis (v0.9.2)

## Implementation Overview

Three-phase widening algorithm in `_analyze_loop_body`:

1. **Phase 1 (Discover)**: Analyze loop body once, collect dimension conflicts
2. **Phase 2 (Stabilize)**: If widening changed anything, re-analyze with widened dims
3. **Phase 3 (Post-loop join)**: Model "loop may not execute" via `widen_env(pre_loop, final)`

## Convergence Guarantee

**Lattice height ≤ 2** for dimensions:
- Bottom: concrete int or symbolic name
- Top: None (unknown)

**Widening jump**: `widen_dim(old, new)` immediately jumps to top (None) on first conflict.

**Why ≤2 iterations?**
- Iteration 1 (Phase 1): Discovers conflicts, widens to None
- Iteration 2 (Phase 2): All widened dims are at top, operations preserve top (e.g., `add_dim(None, x) → None`)
- Fixed point reached in iteration 2

## Semantic Ambiguity: Unknown as Bottom vs Top

**Critical issue**: `Shape.unknown()` has dual semantics:

1. **Unbound variable** (bottom): `Env.get(name)` returns `unknown` when `name not in bindings`
2. **Error/indeterminate shape** (top): Explicit binding `env.set(name, Shape.unknown())`

These are **indistinguishable** at the Shape level.

### widen_shape Unknown Handling

```python
if old.is_unknown():
    return new          # unknown = no info (unbound var) -> adopt new
if new.is_unknown():
    return old          # symmetric
```

**Intended semantics**: unknown as bottom (unbound variable)

**Soundness concern**: If `new.is_unknown()` from explicit binding (error state), returning `old` is unsound:
- Pre-loop: `A = matrix[3 x 3]`
- Post-iteration: `A = unknown` (error in loop body)
- Post-loop join: `widen_shape(matrix[3 x 3], unknown) → matrix[3 x 3]` (unsound!)
- Should return `unknown` to model {matrix[3 x 3], error states}

**Mitigation**: In practice, expressions that produce `unknown` (indexing errors, dimension mismatches) get wrapped in concat/operations that convert `unknown` → `matrix[None x ...]` (via `as_matrix_shape` logic). So the `new.is_unknown()` branch is rarely hit with explicit bindings.

**Test coverage**: No test exercises the unsound case (bound variable → unknown in loop body).

## Post-Loop Join Semantics

Uses `widen_env` (not `join_env`) for post-loop join. This is **correct** because:

1. **For conflicting dims**: `widen_dim(a, b) → None` and `join_dim(a, b) → None` are equivalent
2. **For unknown shapes**: Both treat unknown as bottom symmetrically
3. **Simplicity**: Single operator for widening and post-loop join

Example (loop_may_not_execute.m):
- Pre-loop: `A = matrix[2 x 2]`
- Post-body: `A = matrix[3 x 3]`
- Post-loop: `widen_shape(matrix[2 x 2], matrix[3 x 3]) → matrix[None x None]` ✓

## Validated Behaviors

### Stable Dimensions Preserved
- `widen_dim(3, 3) → 3` (no conflict)
- Test: widen_multiple_vars.m (`B = matrix[4 x 4]` stays concrete)

### Conflicting Dimensions Widened to None
- `widen_dim(3, 4) → None` (conflict)
- Test: widen_col_grows.m (`A = matrix[2 x 3]` → `matrix[2 x 4]` → `matrix[2 x None]` with --fixpoint)

### Scalar ↔ Matrix Kind Conflict → Unknown
- `widen_shape(scalar, matrix[...]) → unknown`
- Test: fixpoint_convergence.m (`A = scalar` → `matrix[2 x 1]` → `unknown` after post-loop join)

### Unbound Variables Adopt Loop Shape
- `widen_shape(unknown, matrix[3 x 3]) → matrix[3 x 3]` (unknown from unbound var)
- Test: loop_exit_join.m (`Y` first assigned in loop → `matrix[3 x 3]`)

### Post-Loop Join Models "May Not Execute"
- `widen_env(pre_loop, post_body)` in Phase 3
- Test: loop_may_not_execute.m (`matrix[2 x 2]` vs `matrix[3 x 3]` → `matrix[None x None]`)

### Symbolic Dimensions Widened Correctly
- Symbolic names (strings) are treated as distinct concrete values
- `widen_dim('n', 'm') → None` (different symbolic names)
- `widen_dim('n', 'n') → 'n'` (same symbolic name)

### Self-Referencing Growth
- Test: widen_self_reference.m (`A = [A, A]` doubles columns each iteration)
- Widening: `widen_dim(3, 6) → None` (column count conflicts)
- Result: `matrix[2 x None]` ✓

### Nested Loops
- Test: nested_loop.m (loop vars `i`, `j` are scalar, used as symbolic dims in `zeros(i, j)`)
- No false conflicts detected ✓

## Remaining Concerns

1. **Unsound unknown handling** (theoretical): `widen_shape(s, unknown) → s` when `unknown` is from explicit binding
   - Not exercised by current tests
   - Would require: loop body assigns variable to unknown (error state), then post-loop join should preserve `unknown`

2. **No widening for `while` loops**: Same algorithm applies (tested in widen_while_growth.m)

3. **Nested loops**: Widening applies to inner loop first, then outer loop sees widened results (test coverage: nested_loop.m)

## Recommendations

- **Add test**: Variable bound to unknown in loop body, check post-loop join preserves unknown
- **Consider refactoring**: Distinguish unbound variables (bottom) from error states (top) at type level
- **Document**: Unknown-as-bottom semantics is intentional for widening, not a bug
