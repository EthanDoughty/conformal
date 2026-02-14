# Implementer Memory

## Key Implementation Patterns

### Builtin Shape Rules Pattern (analysis/analysis_ir.py)
Located in `eval_expr_ir`, Apply handler for `fname in KNOWN_BUILTINS` branch.

**Standard structure**:
```python
if fname == "builtin_name" and len(expr.args) == N:
    try:
        # Extract args via unwrap_arg (raises ValueError for Colon/Range)
        arg_expr = unwrap_arg(expr.args[0])
        # Convert to dim via expr_to_dim_ir, or eval via eval_expr_ir
        result_shape = ...
        return result_shape
    except ValueError:
        # Colon/Range in args: fall through to indexing
        pass
```

**Key helpers**:
- `unwrap_arg(arg)`: IndexArg → Expr (raises ValueError for Colon/Range)
- `expr_to_dim_ir(expr, env)`: Expr → Dim (int | str | None)
- `_eval_index_arg_to_shape(arg, env, warnings)`: IndexArg → Shape

### Shape Rule Categories

**1. Scalar-returning** (det, norm):
```python
if fname in {"det", "norm"} and len(expr.args) == 1:
    _ = _eval_index_arg_to_shape(expr.args[0], env, warnings)
    return Shape.scalar()
```

**2. Pass-through** (abs, sqrt):
```python
if fname in {"abs", "sqrt"} and len(expr.args) == 1:
    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings)
    return arg_shape
```

**3. Dimension swap** (transpose):
```python
if fname == "transpose" and len(expr.args) == 1:
    arg_shape = _eval_index_arg_to_shape(expr.args[0], env, warnings)
    if arg_shape.is_matrix():
        return Shape.matrix(arg_shape.cols, arg_shape.rows)
    return arg_shape
```

**4. Constructor with dim extraction** (zeros, ones, eye, rand, randn):
```python
if fname in {"zeros", "ones"} and len(expr.args) == 1:
    try:
        d = expr_to_dim_ir(unwrap_arg(expr.args[0]), env)
        return shape_of_zeros(d, d)  # or shape_of_ones
    except ValueError:
        pass
```

**5. Shape-dependent dispatch** (diag):
```python
# Only return matrix[n x n] when we can PROVE input is a vector (one dim == 1)
if arg_shape.is_matrix():
    r, c = arg_shape.rows, arg_shape.cols
    if r == 1:
        return Shape.matrix(c, c)  # Row vector → diagonal
    if c == 1:
        return Shape.matrix(r, r)  # Col vector → diagonal
    return Shape.matrix(None, 1)  # Matrix → extract diagonal
```

**6. Squareness check** (inv):
```python
if arg_shape.is_matrix():
    r, c = arg_shape.rows, arg_shape.cols
    if r == c:  # Works for both concrete ints and same symbolic names
        return Shape.matrix(r, c)  # Pass through
    return Shape.unknown()
```

## Critical File Locations

- **Builtin catalog**: `analysis/builtins.py`
  - `KNOWN_BUILTINS`: Functions recognized as calls (not indexing)
  - `BUILTINS_WITH_SHAPE_RULES`: Subset with explicit shape rules
  - After Phase 3: Both sets are equal (19/19)

- **Shape rules**: `analysis/analysis_ir.py` line 193-370 (Apply handler)
- **Test discovery**: `run_all_tests.py` uses `glob("tests/**/*.m", recursive=True)`
- **Test categories**: basics, builtins, control_flow, indexing, literals, loops, recovery, symbolic

## Soundness Principles

1. **Conservative on unknown inputs**: diag(unknown) → unknown, not matrix[None x None]
2. **Prove before refining**: Only return precise shape when we can prove it (e.g., diag vector check)
3. **Silent fallback for unimplemented builtins**: Known builtin without shape rule → unknown (no warning)
4. **Warning for truly unknown functions**: Unbound variable in call position → W_UNKNOWN_FUNCTION

## Common Gotchas

- **Never modify tests** unless TASK.md explicitly requires it
- **Minimal diffs**: Don't refactor "while you're in there"
- **IR analyzer is authoritative**: Legacy analyzer exists only for regression comparison
- **Warning codes use W_* prefix** and must be stable
- **All tests must pass** in both default and --fixpoint modes
