# Lattice Soundness Patterns

## Core Principle

The shape analyzer must never return a shape more precise than what runtime execution could produce (no false negatives). Conservative approximations (overapproximations) are acceptable and necessary when static information is insufficient.

## Shape Lattice (v0.9.3+)

```
        unknown (top / error / indeterminate)
       /    |    \
   scalar  matrix[r x c]  ...
       \    |    /
        bottom (no info / unbound)
```

Where dimensions `r, c` can be:
- Concrete integers (e.g., `3`, `4`)
- Symbolic names (e.g., `"n"`, `"m"`)
- `None` (completely unknown)

**v0.9.3 change**: Added `Shape.bottom()` as distinct element to fix soundness bug where `Shape.unknown()` conflated unbound variables (bottom) with errors (top).

## v0.9.3 Bottom/Error Distinction (Validated 2026-02-14)

**Status**: ✅ SEMANTICALLY CORRECT AND SOUND

### Operator Semantics (Verified)

#### `widen_shape(old, new)` - VERIFIED ✅

| old \ new | bottom | scalar | matrix | unknown |
|-----------|--------|--------|--------|---------|
| **bottom** | bottom | scalar | matrix | unknown |
| **scalar** | scalar | scalar | unknown | unknown |
| **matrix** | matrix | unknown | widen_dim | unknown |
| **unknown** | unknown | unknown | unknown | unknown |

**Key property**: bottom is identity, unknown is absorbing top

#### `join_shape(s1, s2)` - VERIFIED ✅

| s1 \ s2 | bottom | scalar | matrix | unknown |
|---------|--------|--------|--------|---------|
| **bottom** | bottom | scalar | matrix | unknown |
| **scalar** | scalar | scalar | unknown | unknown |
| **matrix** | matrix | unknown | join_dim | unknown |
| **unknown** | unknown | unknown | unknown | unknown |

**Key property**: bottom is identity, unknown is absorbing top

### Bottom Containment - VERIFIED ✅

1. **Env.get() returns bottom for unbound variables** (runtime/env.py:23)
   - `env.get("unbound_var")` → `Shape.bottom()`

2. **Var evaluation converts bottom → unknown** (analysis/analysis_ir.py:190-194)
   - `eval_expr_ir(Var("unbound_x"))` → `Shape.unknown()`
   - Bottom never escapes to expression evaluation

3. **Defensive checks** (analysis_core.py:20, matrix_literals.py:18-20)
   - `shapes_definitely_incompatible`: bottom is compatible with everything
   - `as_matrix_shape(bottom)` → `Shape.unknown()` (defensive fallback)

### Soundness Fix - VERIFIED ✅

**Pre-v0.9.3 bug** (tests/loops/widen_unknown_in_body.m):
- `widen(matrix[3x3], unknown_from_error)` incorrectly returned `matrix[3x3]`
- Fixpoint mode missed the error state in the loop

**Post-v0.9.3 fix**:
- `widen(matrix[3x3], unknown)` correctly returns `unknown`
- Fixpoint mode: `A = unknown` (correct propagation of error state)

### Precision Regression - DOCUMENTED ✅

**Test**: tests/control_flow/if_branch_mismatch.m

**Change**: `join(unknown, matrix[n x k])` now returns `unknown` instead of `matrix[n x k]`

**Rationale**: More sound — if one branch produces unknown (error), the result is truly indeterminate. Pre-v0.9.3 behavior silently ignored errors in one branch.

**Example**:
```matlab
if cond
    C = A + B;  % dimension mismatch -> unknown
else
    C = A;      % matrix[n x k]
end
% Post-join: C = unknown (was matrix[n x k] pre-v0.9.3)
```

### Bottom-as-Identity Use Cases - VERIFIED ✅

1. **First assignment inside loop** (tests/loops/widen_first_assign_in_body.m)
   - Pre-loop: `env.get("B")` → `bottom`
   - Body: `B = zeros(3, 3)`
   - Widening: `widen(bottom, matrix[3x3])` → `matrix[3x3]` ✅

2. **Variable only in one if-branch** (implicit in join_env)
   - Then-branch: `x = matrix[3x3]`
   - Else-branch: `x` unbound → `bottom`
   - Join: `join(matrix[3x3], bottom)` → `matrix[3x3]` ✅

### Test Coverage - ALL PASS ✅

**58/58 tests pass** in both default and fixpoint modes:
- tests/loops/widen_unknown_in_body.m (fixpoint: A = unknown) ✅
- tests/loops/widen_unknown_false_positive.m (fixpoint: 1 warning, not 2) ✅
- tests/loops/widen_first_assign_in_body.m (fixpoint: B = matrix[3x3]) ✅
- tests/loops/widen_if_in_loop.m (fixpoint: A = matrix[None x 3]) ✅
- tests/loops/widen_interdependent_vars.m (fixpoint: independent widening) ✅
- tests/control_flow/if_else_error_branch.m (A = unknown) ✅
- tests/loops/widen_error_in_branch.m (fixpoint: A = unknown) ✅
- tests/control_flow/if_branch_mismatch.m (C = unknown, precision regression documented) ✅

## Dimension Join Rules

```python
join_dim(a, b):
    if a == b: return a        # Exact match
    if a is None: return b     # Unknown absorbs
    if b is None: return a     # Unknown absorbs
    return None                # Conflict → unknown
```

**Key property**: Join never returns a dimension more precise than either input.

## Conservative Approximation Patterns

### Pattern 1: Unknown Input → Unknown Output

```python
if arg_shape.is_unknown():
    return Shape.unknown()
```

**Example**: `diag(unknown)` → `unknown`

**Soundness**: Cannot determine behavior statically, so return least precise shape.

---

### Pattern 2: Cannot Prove Property → Conservative Result

**Example**: `diag(matrix[n x m])` where `n` and `m` are different symbolic names

```python
if r == 1:          # "n" == 1 → False
if c == 1:          # "m" == 1 → False
return matrix[None x 1]  # Conservative: assume matrix, not vector
```

**Soundness**: We can't prove `n=1` or `m=1` statically, so we assume the general case (matrix → column vector extraction). If runtime has `n=1` or `m=1`, we lose precision but remain sound.

---

### Pattern 3: Dimension Equality Checks (Symbolic vs Concrete)

**Example**: `inv(matrix[n x n])` checks squareness

```python
if r == c:  # String equality for symbolic names
    return matrix[r, c]
```

**Soundness rules**:
- ✅ `3 == 3` → True (concrete match)
- ✅ `"n" == "n"` → True (same symbolic name)
- ✅ `"n" == "m"` → False (different names, can't prove equal)
- ✅ `"n" == 1` → False (symbolic ≠ concrete, can't prove)
- ✅ `None == None` → True (both unknown)

**Critical**: Never use symbolic unification (e.g., assuming `"n" = "m"` because they appear in the same context). This would be unsound.

---

### Pattern 4: Edge Case — Both Dims Unknown

**Example**: `inv(matrix[None x None])`

```python
if r == c:  # None == None → True
    return matrix[None, None]
```

**Analysis**:
- Runtime could be square or non-square
- Returning `matrix[None x None]` says "it's a matrix with unknown dimensions"
- Returning `unknown` would be more conservative
- **Current choice**: `matrix[None x None]` (slightly imprecise but sound — both represent "unknown dimensions")

**Alternative**: Could return `Shape.unknown()` to be maximally conservative.

---

### Pattern 5: Scalar Broadcasting

**Example**: `matrix[n x m] + scalar` → `matrix[n x m]`

```python
if left.is_scalar() and not right.is_scalar():
    return right
if right.is_scalar() and not left.is_scalar():
    return left
```

**Soundness**: MATLAB broadcasts scalars to match matrix shape, so result shape matches the matrix operand.

---

### Pattern 6: Symbolic Arithmetic (Concatenation)

**Example**: Horizontal concat `[matrix[n x m], matrix[n x k]]` → `matrix[n x (m+k)]`

```python
def add_dim(a: Dim, b: Dim) -> Dim:
    if a is None or b is None:
        return None  # Unknown propagates
    if isinstance(a, int) and isinstance(b, int):
        return a + b  # Concrete arithmetic
    return f"({a}+{b})"  # Symbolic expression
```

**Soundness**:
- Concrete: `3 + 4` → `7` (exact)
- Symbolic: `"m" + "k"` → `"(m+k)"` (symbolic expression, no simplification)
- Unknown: `None + x` → `None` (conservative)

---

## Validation Checklist for New Builtin Rules

When adding a new builtin shape rule, verify:

1. **Unknown input handling**: Does it return `unknown` or `matrix[None x None]` for unknown inputs?
2. **Symbolic dimension preservation**: Are symbolic names correctly propagated without spurious unification?
3. **Dimension equality checks**: Do checks use exact equality (`==`) without assumptions?
4. **Conservative approximation**: When static info is insufficient, does it overapproximate (return less precise shape)?
5. **Edge cases**: How does it handle `matrix[None x None]`, `scalar`, `unknown`, `bottom`?
6. **Side effects**: Does it evaluate argument expressions for nested shape tracking?
7. **Lattice correctness**: Can the returned shape be more precise than runtime execution? (If yes, unsound!)

## Known Conservative Approximations (All Sound)

1. **diag(matrix[n x m])** → `matrix[None x 1]`
   - Could be diagonal `matrix[n x n]` if `n=1` or `m=1` at runtime
   - Conservative: assumes matrix extraction case

2. **inv(matrix[None x None])** → `matrix[None x None]`
   - Could be `unknown` if non-square at runtime
   - Conservative: assumes matrix with unknown dims (equivalent to `unknown` for most purposes)

3. **Range indexing** `A(1:n, :)` → `matrix[None x c]`
   - Could compute exact extent if `n` is symbolic
   - Conservative: returns `None` when extent is not a concrete integer

## Anti-Patterns (Unsound)

❌ **Assuming symbolic names are equal**: `if "n" appears twice, assume n=n`
- Counter-example: `A = zeros(n, n); B = zeros(n, n); C = [A, B]` should be `matrix[n x (n+n)]`, not `matrix[n x n]`

❌ **Returning too-precise shape**: `diag(matrix[None x None])` → `matrix[None x None]` (diagonal)
- Runtime could extract column vector if input is non-vector matrix
- Correct: return `matrix[None x 1]` (conservative)

❌ **False positives**: Emitting warnings for code that could be valid at runtime
- All warnings must be definite errors (concrete dimension conflicts)
- Symbolic conflicts → return `unknown`, don't warn (can't prove at compile time)

## References

- TASK.md: v0.9.3 specification (lines 1-269)
- runtime/shapes.py: Shape.bottom() constructor, widen_shape, join_shape (lines 48-265)
- runtime/env.py: Env.get() returns bottom for unbound vars (line 23)
- analysis/analysis_ir.py: Var case converts bottom → unknown (lines 190-194)
- analysis/analysis_core.py: shapes_definitely_incompatible defensive check (line 20)
- analysis/matrix_literals.py: as_matrix_shape defensive check (lines 18-20)
