# Task: Phase 3 — Complete Builtin Shape Rules (v0.9.0)

## Goal
Implement shape rules for all remaining builtins that currently return `unknown`, plus fill
the `zeros(n)`/`ones(n)` 1-arg gap. After this, every builtin in `KNOWN_BUILTINS` has a
precise shape rule. Milestone version: **0.9.0**.

## Scope

### A. New shape rules (5 builtins)

| Builtin | Signature(s) | Output shape |
|---------|-------------|-------------|
| `det` | `det(A)` | `scalar` |
| `diag` | `diag(v)` where v is vector (`n x 1` or `1 x n`) | `matrix[n x n]` |
| `diag` | `diag(A)` where A is matrix (`m x n`, neither dim is 1) | `matrix[None x 1]` (can't compute `min(m,n)` symbolically) |
| `diag` | `diag(A)` where A is `unknown` or `matrix[None x None]` | `unknown` (can't distinguish vector from matrix) |
| `inv` | `inv(A)` where A is square (`n x n`) | `matrix[n x n]` (pass-through) |
| `inv` | `inv(A)` where A is non-square (`m x n`, m != n) | `unknown` (mathematically undefined) |
| `linspace` | `linspace(a, b)` | `matrix[1 x 100]` (MATLAB default) |
| `linspace` | `linspace(a, b, n)` | `matrix[1 x n]` (n via `expr_to_dim_ir`) |
| `norm` | `norm(x)` | `scalar` |

### B. Fix existing gap (2 builtins)

| Builtin | Signature | Current behavior | Fixed behavior |
|---------|-----------|-----------------|---------------|
| `zeros` | `zeros(n)` | Falls through to `unknown` | `matrix[n x n]` (mirrors `eye(n)`) |
| `ones` | `ones(n)` | Falls through to `unknown` | `matrix[n x n]` (mirrors `eye(n)`) |

### C. Update registry
- Add `det`, `diag`, `inv`, `linspace`, `norm` to `BUILTINS_WITH_SHAPE_RULES` in `analysis/builtins.py`
- After this, `BUILTINS_WITH_SHAPE_RULES == KNOWN_BUILTINS` (19/19)

## Non-goals
- `zeros([m, n])` array syntax (requires `MatrixLit` arg inspection — different code path, <5% usage)
- 3D arrays (`zeros(m, n, p)`) — shape domain is 2D only; existing fall-through to `unknown` is fine
- Refactoring `unwrap_arg`/`expr_to_dim_ir` pattern (optional nice-to-have, not required)
- Extending `expr_to_dim_ir` to handle `BinOp` expressions like `n+1` (future work)
- Parser, IR, or lowering changes (pure analysis-layer work)

## Implementation Notes

### diag — shape-dependent dispatch
This is the trickiest rule. Decision tree:
1. Input is `scalar` → `matrix[1 x 1]`
2. Input is `matrix[n x 1]` or `matrix[1 x n]` (one dim is concretely 1) → `matrix[n x n]`
3. Input is `matrix[m x n]` where both dims are known and equal → could be either; treat as matrix extraction → `matrix[None x 1]`
4. Input is `matrix[m x n]` where both dims are known and different → matrix extraction → `matrix[None x 1]`
5. Input is `unknown` or `matrix[None x None]` → `unknown` (can't distinguish vector from matrix)

Key soundness requirement: only dispatch to "vector → diagonal matrix" when we can **prove** the input is a vector (one dimension is concretely 1).

### inv — pass-through with squareness check
- If both dims are known and equal: pass through shape
- If both dims are symbolic and identical (same symbol): pass through shape
- If dims are known and different: return `unknown` (mathematically undefined, but no warning — could be runtime-determined)
- If dims are unknown: return `unknown`

### zeros(n) / ones(n) — 1-arg constructor form
Add `elif len(expr.args) == 1` branch above the existing 2-arg branch. Same pattern as `eye(n)`:
extract dim via `expr_to_dim_ir`, return `matrix[dim x dim]`.

### Optional refactor (nice-to-have)
The `zeros`/`ones`/`eye`/`rand`/`randn` family all share identical 1-arg/2-arg dispatch logic.
A `_builtin_constructor_shape(args, env)` helper could reduce repetition. Only do this if the
implementer finds the code getting unwieldy — not required for correctness.

## Invariants
- All existing 43 tests pass in both default and `--fixpoint` modes
- `BUILTINS_WITH_SHAPE_RULES` becomes equal to `KNOWN_BUILTINS` (19/19)
- Apply remains the sole call/index IR node
- No parser, IR, or lowering changes

## Acceptance Criteria
- [ ] `det(A)` → `scalar`
- [ ] `diag(v)` for vector → `matrix[n x n]`
- [ ] `diag(A)` for matrix → `matrix[None x 1]`
- [ ] `diag(unknown)` → `unknown` (soundness)
- [ ] `inv(A)` for square matrix → pass-through shape
- [ ] `inv(A)` for non-square/unknown → `unknown`
- [ ] `linspace(a, b)` → `matrix[1 x 100]`
- [ ] `linspace(a, b, n)` → `matrix[1 x n]`
- [ ] `norm(x)` → `scalar`
- [ ] `zeros(n)` → `matrix[n x n]` (was `unknown`)
- [ ] `ones(n)` → `matrix[n x n]` (was `unknown`)
- [ ] All 5 new builtins in `BUILTINS_WITH_SHAPE_RULES`
- [ ] New test file `tests/builtins/remaining_builtins.m` passes
- [ ] All existing tests pass: `python3 mmshape.py --tests` and `python3 mmshape.py --fixpoint --tests`

## Test File: `tests/builtins/remaining_builtins.m`

```matlab
% Test: Phase 3 builtin shape rules
% EXPECT: warnings = 0

% --- det ---
A1 = eye(3);
det_concrete = det(A1);
% EXPECT: det_concrete = scalar
det_symbolic = det(zeros(n, n));
% EXPECT: det_symbolic = scalar

% --- diag (vector → diagonal matrix) ---
v1 = zeros(5, 1);
diag_vec_concrete = diag(v1);
% EXPECT: diag_vec_concrete = matrix[5 x 5]
v2 = zeros(n, 1);
diag_vec_symbolic = diag(v2);
% EXPECT: diag_vec_symbolic = matrix[n x n]
v3 = zeros(1, k);
diag_row_vec = diag(v3);
% EXPECT: diag_row_vec = matrix[k x k]

% --- diag (matrix → column vector) ---
M1 = zeros(3, 4);
diag_mat = diag(M1);
% EXPECT: diag_mat = matrix[None x 1]

% --- inv ---
inv_concrete = inv(eye(3));
% EXPECT: inv_concrete = matrix[3 x 3]
inv_symbolic = inv(zeros(n, n));
% EXPECT: inv_symbolic = matrix[n x n]

% --- linspace ---
ls_default = linspace(0, 1);
% EXPECT: ls_default = matrix[1 x 100]
ls_concrete = linspace(0, 10, 50);
% EXPECT: ls_concrete = matrix[1 x 50]
ls_symbolic = linspace(0, 1, n);
% EXPECT: ls_symbolic = matrix[1 x n]

% --- norm ---
norm_vec = norm(v1);
% EXPECT: norm_vec = scalar
norm_mat = norm(A1);
% EXPECT: norm_mat = scalar

% --- zeros/ones 1-arg form (gap fix) ---
z_sq = zeros(4);
% EXPECT: z_sq = matrix[4 x 4]
z_sq_sym = zeros(n);
% EXPECT: z_sq_sym = matrix[n x n]
o_sq = ones(3);
% EXPECT: o_sq = matrix[3 x 3]
o_sq_sym = ones(n);
% EXPECT: o_sq_sym = matrix[n x n]
```

## Commands
```bash
python3 mmshape.py tests/builtins/remaining_builtins.m   # new test
python3 mmshape.py --tests                                # full suite
python3 mmshape.py --fixpoint --tests                     # fixpoint mode
```
