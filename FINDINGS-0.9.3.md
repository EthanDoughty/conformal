# FINDINGS-0.9.3: Fix Bottom/Error Ambiguity in Shape Lattice

## Problem

`Shape.unknown()` serves two roles:
1. **Unbound variable** (bottom): `Env.get()` returns unknown for variables not yet assigned
2. **Error/indeterminate** (top): Operations that fail return unknown

Both `widen_shape` and `join_shape` treat unknown as bottom. This causes a confirmed soundness bug:

```
widen_shape(matrix[3x3], unknown_from_error) → matrix[3x3]  // WRONG
```

**Bug location**: `runtime/shapes.py:199-202`
**Confirmed test**: `tests/loops/widen_unknown_in_body.m` — fixpoint gives `matrix[3x3]` instead of `unknown`
**False positive**: Downstream `A * zeros(5,5)` spuriously warns about inner dim 3≠5 in fixpoint mode

## Fix: Option A — `Shape.bottom()`

Add a new lattice element. Standard approach used by Astree, IKOS, Infer.

```
        unknown (top / error / indeterminate)
       /    |    \
   scalar  matrix[r x c]  ...
       \    |    /
        bottom (no info / unbound)
```

## Design Decisions (User Approved)

| # | Question | Decision |
|---|----------|----------|
| 1 | Scope | Fix bottom/error AND add edge case tests in 0.9.3 |
| 2 | `join_shape` precision | Sound: `join(unknown, X) = unknown`. Note precision regression. |
| 3 | `Env.get()` unbound | Return `Shape.bottom()` |
| 4 | Break/continue | Defer — parser doesn't support it yet |
| 5 | User-defined functions | Keep in mind for v0.10.0; bottom/error distinction helps there |
| 6 | `eval_expr_ir` fallback | Treat as error-unknown, emit message that shape is unknown |
| 7 | Nested interdependencies | Test in test suite |

## Operator Semantics

### `widen_shape(old, new)`

| old \ new | bottom | scalar | matrix | unknown |
|-----------|--------|--------|--------|---------|
| **bottom** | bottom | scalar | matrix | unknown |
| **scalar** | scalar | scalar | unknown | unknown |
| **matrix** | matrix | unknown | widen_dim | unknown |
| **unknown** | unknown | unknown | unknown | unknown |

### `join_shape(s1, s2)`

| s1 \ s2 | bottom | scalar | matrix | unknown |
|----------|--------|--------|--------|---------|
| **bottom** | bottom | scalar | matrix | unknown |
| **scalar** | scalar | scalar | unknown | unknown |
| **matrix** | matrix | unknown | join_dim | unknown |
| **unknown** | unknown | unknown | unknown | unknown |

**Key**: bottom is identity (absorbing nothing). unknown is top (absorbing everything).

## Files to Change

### `runtime/shapes.py`
- Add `Shape.bottom()` constructor: `Shape(kind="bottom")`
- Add `is_bottom()` predicate
- Add `__str__` case for bottom (internal only)
- Update `widen_shape` (lines 199-210): bottom is identity, unknown is absorbing top
- Update `join_shape` (lines 217-231): bottom is identity, unknown is absorbing top

### `runtime/env.py`
- `Env.get()` (line 23): return `Shape.bottom()` instead of `Shape.unknown()`

### `analysis/analysis_ir.py`
- `eval_expr_ir` Var case (line 190-191): convert bottom → unknown for expression eval
  ```python
  if isinstance(expr, Var):
      shape = env.get(expr.name)
      return shape if not shape.is_bottom() else Shape.unknown()
  ```
- This keeps bottom contained to Env-level operators; expression eval never sees bottom

### `analysis/analysis_core.py`
- `shapes_definitely_incompatible` (line 19): add `or old.is_bottom() or new.is_bottom()` to the early-return
  (bottom is compatible with everything, same as current unknown behavior)

### `analysis/matrix_literals.py`
- `as_matrix_shape` (line 10): add safety net for bottom → return unknown
  (should never be reached, but defensive)

### Tests
- Update `tests/loops/widen_unknown_in_body.m`: change EXPECT_FIXPOINT from `A = matrix[3 x 3]` to `A = unknown`

## Edge Case Tests to Add

### 1. `tests/loops/widen_unknown_false_positive.m`
Demonstrates user-visible false positive from the pre-fix bug.
```matlab
% Test: Unknown function in loop + downstream matmul (false positive before fix)
% Pre-fix: fixpoint preserves matrix[3x3], causing spurious inner-dim warning on A*zeros(5,5)
% Post-fix: A = unknown after loop, no spurious warning
% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT: B = unknown
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = unknown
% EXPECT_FIXPOINT: B = unknown

A = zeros(3, 3);
for i = 1:n
    A = unknown_func();
end
B = A * zeros(5, 5);
```
Current behavior (BUGGY): fixpoint gives 2 warnings, A=matrix[3x3], B=unknown.
Expected after fix: 1 warning (W_UNKNOWN_FUNCTION only), A=unknown, B=unknown.

### 2. `tests/loops/widen_if_in_loop.m`
If-inside-loop with partial assignment (one branch grows, other keeps old value).
```matlab
% Test: Conditional growth inside loop body
% Only one branch of if modifies A; else keeps old value
% The if-join already widens, then loop widening preserves it
% EXPECT: warnings = 1
% EXPECT: A = matrix[None x 3]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x 3]

A = zeros(3, 3);
for i = 1:n
    if cond
        A = [A; zeros(1, 3)];
    end
end
```

### 3. `tests/loops/widen_first_assign_in_body.m`
Variable first assigned inside loop body (unbound pre-loop → bottom as identity).
```matlab
% Test: Variable first assigned inside loop body
% B is unbound pre-loop; bottom-as-identity means B gets the loop body's shape
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 3]
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[3 x 3]

for i = 1:n
    B = zeros(3, 3);
end
```

### 4. `tests/loops/widen_interdependent_vars.m`
Two variables with independent growth axes in same loop.
```matlab
% Test: Two variables growing on different axes in same loop
% A grows rows, B grows cols — each should widen independently
% EXPECT: warnings = 2
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT_FIXPOINT: warnings = 2
% EXPECT_FIXPOINT: A = matrix[None x 3]
% EXPECT_FIXPOINT: B = matrix[3 x None]

A = zeros(2, 3);
B = zeros(3, 2);
for i = 1:n
    A = [A; zeros(1, 3)];
    B = [B, zeros(3, 1)];
end
```

### 5. `tests/control_flow/if_else_error_branch.m`
If-else where one branch returns unknown (precision regression test).
```matlab
% Test: If-else where one branch calls unknown function
% Sound behavior: join(matrix[3x3], unknown) = unknown
% This is a precision regression from pre-0.9.3 (was matrix[3x3])
% but is the correct sound choice
% EXPECT: warnings = 1
% EXPECT: A = unknown

if cond
    A = zeros(3, 3);
else
    A = unknown_func();
end
```

### 6. `tests/loops/widen_error_in_branch.m`
Error in one branch of if-else inside a loop body.
```matlab
% Test: One loop branch calls unknown function, other grows matrix
% Sound behavior: if-join gives unknown (error absorbs), then widening sees unknown
% EXPECT: warnings = 2
% EXPECT: A = matrix[4 x 3]
% EXPECT_FIXPOINT: warnings = 2
% EXPECT_FIXPOINT: A = unknown

A = zeros(3, 3);
for i = 1:n
    if cond
        A = unknown_func();
    else
        A = [A; zeros(1, 3)];
    end
end
```
Current behavior (BUGGY): fixpoint gives `matrix[None x 3]` (because join in the if-else treats unknown as bottom).
Expected after fix: `A = unknown` in fixpoint mode.

## Precision Regression Note

After this fix, `join_shape(unknown, X) = unknown` instead of `X`. This means if-else branches where one path produces an error will propagate unknown to the merged result. This is **more sound** but **less precise**.

**Impact on existing tests**: Zero. Audited all 52 tests — no existing test has an unknown function call inside an if-else branch. The only tests with if-else use concrete shapes (`if_branch_mismatch.m`, `suspicious_comparison.m`).

**New test documenting this**: `tests/control_flow/if_else_error_branch.m`

## Deferred Items

### Break/Continue (defer to parser work)
MATLAB supports `break` and `continue` in loops. These affect which variables get assigned (a `break` path may leave variables at pre-iteration shapes). Currently not parsed. When parser support is added, loop analysis must model early exit paths in the post-loop join.

### Narrowing (defer indefinitely)
Classical narrowing refines post-widening approximations. Not useful here because dimension lattice has height 2 — Phase 2 (Stabilize) already acts as de facto narrowing. Would only help if dimension domain gains interval tracking (e.g., "rows in [3, 10]").

### User-Defined Functions (v0.10.0)
The bottom/error distinction directly benefits interprocedural analysis: "function not yet analyzed" (bottom) vs "function body has unresolvable shape" (error-unknown) must be distinguished for interprocedural fixpoint iteration.

## Risks

1. **Bottom leaking into expressions**: Mitigated by converting bottom → unknown in `eval_expr_ir` Var case. Expression eval code (builtins, binops, indexing) needs zero changes.

2. **`__eq__` distinguishes bottom from unknown**: `Shape(kind="bottom") != Shape(kind="unknown")`. This is correct and desired for `widen_env` equality checks (`widened.bindings != env.bindings`).

3. **`as_matrix_shape(bottom)`**: Should never happen (bottom is converted before reaching concat logic), but add defensive return of `unknown` just in case.

4. **Env printing**: `Env.__repr__` iterates bindings. Bottom should never appear in bindings (only returned by `get()` for missing keys), so no format change needed.

## Estimated Diff

~40 lines across 5 files + 6 new test files + 1 test update. Surgical change.
