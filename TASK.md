# Task: Extend expr_to_dim_ir to Support Arithmetic Expressions

## Goal
Enable `expr_to_dim_ir` in `analysis/analysis_ir.py` to extract symbolic dimensions from arithmetic expressions (`n+1`, `2*m`, `k+m+1`), allowing constructors like `zeros(n+1, m)` and `reshape(A, 2*n, k)` to produce precise symbolic shapes instead of `matrix[None x ...]`.

## Scope
- Modify `expr_to_dim_ir` function (lines 507-524 of `analysis/analysis_ir.py`) to handle `BinOp` nodes
- Support operators: `+`, `-`, `*` (minimum viable set)
- Recurse into BinOp operands to build composite symbolic dimensions
- Use existing `add_dim` and `mul_dim` from `runtime/shapes.py`

## Non-goals
- No changes to shape domain (`runtime/shapes.py`)
- No new dimension arithmetic functions (add_dim/mul_dim already exist)
- No support for division (not useful for dimensions)
- No changes to builtin handlers (they already call `expr_to_dim_ir`)
- No changes to test infrastructure

## Invariants Impacted
- IR analyzer authoritative: Preserved (only extending expr_to_dim_ir)
- All tests pass: Must preserve (44 existing tests continue passing)
- Shape domain operations: Relies on existing add_dim/mul_dim behavior

## Acceptance Criteria
- [ ] `zeros(n+1, m)` produces `matrix[(n+1) x m]` not `matrix[None x m]`
- [ ] `zeros(2*n, 3)` produces `matrix[(2*n) x 3]` not `matrix[None x 3]`
- [ ] `reshape(A, n+m, k)` produces `matrix[(n+m) x k]` not `matrix[None x k]`
- [ ] `zeros(n+m+k, 1)` produces `matrix[(n+(m+k)) x 1]` (nested addition)
- [ ] `zeros(length(A), m)` produces `matrix[None x m]` (function call returns None)
- [ ] `zeros(n-1, m)` produces `matrix[(n+-1) x m]` (subtraction via add_dim with negative)
- [ ] All existing tests pass: `python3 mmshape.py --tests`

## Commands to Run
```
python3 mmshape.py tests/builtins/dim_arithmetic.m
python3 mmshape.py --tests
```

## Tests to Add/Change
- `tests/builtins/dim_arithmetic.m`: New test file covering:
  - Simple addition: `zeros(n+1, m)` → `matrix[(n+1) x m]`
  - Simple multiplication: `zeros(2*n, k)` → `matrix[(2*n) x k]`
  - Nested addition: `zeros(n+m+k, 1)` → `matrix[(n+(m+k)) x 1]`
  - Subtraction: `zeros(n-1, m)` → `matrix[(n+-1) x m]`
  - Mixed symbolic/concrete: `zeros(n+2, 3*m)` → `matrix[(n+2) x (3*m)]`
  - reshape with arithmetic: `reshape(A, 2*n, m+1)` → `matrix[(2*n) x (m+1)]`
  - Edge case: `zeros(length(A), m)` → `matrix[None x m]` (nested call fails gracefully)
  - Expected warnings: 0
