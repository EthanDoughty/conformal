# Lattice Soundness Patterns

## Core Principle

The shape analyzer must never return a shape more precise than what runtime execution could produce (no false negatives). Conservative approximations (overapproximations) are acceptable and necessary when static information is insufficient.

## Shape Lattice

```
        unknown
       /       \
  scalar    matrix[r x c]
```

Where dimensions `r, c` can be:
- Concrete integers (e.g., `3`, `4`)
- Symbolic names (e.g., `"n"`, `"m"`)
- `None` (completely unknown)

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
5. **Edge cases**: How does it handle `matrix[None x None]`, `scalar`, `unknown`?
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
