# Task: Rich Builtin Shape Rules (Phase 3)

## Goal
Add shape inference rules for MATLAB builtin functions so the analyzer produces concrete shapes instead of `unknown` for common matrix constructors, element-wise operations, and query functions.

## Scope
- Add shape rules for 4 builtin categories in `analysis/analysis_ir.py`:
  1. **Matrix constructors** (0/1/2-arg forms): `eye`, `rand`, `randn`
     - 0 args: `rand()` → scalar, `randn()` → scalar
     - 1 arg: `eye(n)` → n×n, `rand(n)` → n×n, `randn(n)` → n×n
     - 2 args: `eye(m,n)` → m×n, `rand(m,n)` → m×n, `randn(m,n)` → m×n
     - (Same signature pattern as existing `zeros`/`ones`)
  2. **Unary element-wise**: `abs`, `sqrt` (output shape = input shape)
  3. **Transpose function**: `transpose` (swaps row/col dimensions — NOT shape-preserving)
  4. **Query functions**: `length`, `numel` (return scalar)
- Factor out shared try/except unwrap boilerplate into a helper (e.g., `_eval_builtin_args`)
- Update `_BUILTINS_WITH_SHAPE_RULES` set to include all newly handled builtins
- Add test file exercising all new rules with symbolic and concrete dimensions

## Non-goals
- Reshaping functions (`reshape`, `repmat`) — too complex, defer
- Reduction functions (`sum`, `min`, `max`) — dim argument handling complex, defer
- Advanced functions (`diag`, `inv`, `det`, `trace`, `norm`, `linspace`) — defer to future phase
- Three-dimensional arrays (e.g., `zeros(2,3,4)`) — not supported by shape domain

## Invariants Impacted
- **IR analyzer authoritative**: All changes in `analysis_ir.py` (no legacy analyzer changes)
- **Test expectations reflect IR analyzer**: New test expectations must match IR analyzer output
- **All existing tests pass**: No regressions in existing test suite
- **Known builtins return shape or unknown**: Every builtin in `KNOWN_BUILTINS` must either have a rule or return `unknown` silently
- **Warning stability**: No new warning codes introduced
- **`transpose()` and `.'` must agree**: Both must swap dimensions, share logic or produce identical results

## Acceptance Criteria
- [ ] **Constructors (0-arg)**: `rand()` → `scalar`, `randn()` → `scalar`
- [ ] **Constructors (1-arg)**: `eye(n)` → `matrix[n x n]`, `rand(n)` → `matrix[n x n]`, `randn(n)` → `matrix[n x n]`
- [ ] **Constructors (2-arg)**: `eye(m,n)` → `matrix[m x n]`, `rand(2,3)` → `matrix[2 x 3]`, `randn(k,2)` → `matrix[k x 2]`
- [ ] **Element-wise**: `abs(A)` → same shape as `A`, `sqrt(B)` → same shape as `B` (including scalar, matrix, unknown pass-through)
- [ ] **Transpose function**: `transpose(C)` → swapped dimensions (row↔col), consistent with `.'` operator
- [ ] **Queries**: `length(v)` → `scalar`, `numel(M)` → `scalar`
- [ ] **Symbolic dimensions**: `eye(n)` → `matrix[n x n]` where `n` is symbolic
- [ ] **Edge case**: `eye(0)` → `matrix[0 x 0]` (empty matrix)
- [ ] **Unwrap helper**: Shared `_eval_builtin_args` or similar reduces try/except boilerplate
- [ ] `_BUILTINS_WITH_SHAPE_RULES` updated to include all newly handled builtins
- [ ] All existing tests pass: `python3 mmshape.py --tests`
- [ ] New test file validates all new shape rules

## Commands to Run
```bash
# Run all tests including new one
python3 mmshape.py --tests

# Run new test individually
python3 mmshape.py tests/test31.m
```

## Tests to Add/Change
- **test31.m**: Rich builtin shape rules
  - Constructors 0-arg: `rand()`, `randn()`
  - Constructors 1-arg: `eye(n)`, `rand(n)`, `randn(n)` (should produce n×n, NOT n×1)
  - Constructors 2-arg: `eye(m,n)`, `rand(2,3)`, `randn(k,2)`
  - Element-wise: `abs(zeros(2,3))`, `sqrt(ones(n,m))`
  - Transpose function: `transpose(zeros(2,3))` → `matrix[3 x 2]` (must match `.'` operator)
  - Queries: `length(v)`, `numel(M)`
  - Edge case: `eye(0)` → `matrix[0 x 0]`
  - Expected warnings: 0 (all functions recognized and have shape rules)
  - Assertions: Use `% EXPECT:` format for all variable shapes and warning count

## Implementation Notes

**Constructor pattern** (shared for eye/rand/randn/zeros/ones):
```python
if len(args) == 0:
    return Shape.scalar()
elif len(args) == 1:
    d = expr_to_dim_ir(args[0], env)
    return Shape.matrix(d, d)  # n×n square matrix
elif len(args) == 2:
    r = expr_to_dim_ir(args[0], env)
    c = expr_to_dim_ir(args[1], env)
    return Shape.matrix(r, c)
else:
    return Shape.unknown()  # 3D+ not supported
```

**Transpose function** — reuse logic from `Transpose` IR node handler (lines ~226-230):
```python
# Same swap logic as .' operator
if shape.kind == "matrix":
    return Shape.matrix(shape.cols, shape.rows)
elif shape.kind == "scalar":
    return Shape.scalar()
else:
    return Shape.unknown()
```

**Element-wise pattern** (abs, sqrt): evaluate arg shape, return it unchanged.

**Query pattern** (length, numel): evaluate arg (to trigger any nested warnings), return `scalar`.

**Unwrap helper** to reduce boilerplate:
```python
def _eval_builtin_args(expr, env, warnings):
    """Unwrap Apply args and evaluate shapes. Returns None if any arg is Colon/Range."""
    exprs = []
    for arg in expr.args:
        try:
            exprs.append(unwrap_arg(arg))
        except ValueError:
            return None  # Colon/Range → fall through to indexing
    return exprs
```
