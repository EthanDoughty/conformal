# Spec Writer Memory

## Project Structure
- **IR analyzer is authoritative**: `analysis/analysis_ir.py` is the source of truth
- **Legacy analyzer**: `analysis/analysis_legacy.py` exists only for regression comparison
- **Test format**: Inline `% EXPECT:` assertions in .m files in `tests/` directory
- **Test discovery**: Dynamic via `glob("tests/test*.m")` in `run_all_tests.py`
- **Currently 28 tests**: All passing as of 2026-02-11

## Control Flow Current Implementation
- Lines 79-93 of `analysis_ir.py`: If statements handled by:
  1. Evaluate condition (line 80)
  2. Copy environment for then/else branches (lines 82-83)
  3. Analyze each branch independently (lines 85-88)
  4. Join environments using `join_env` from `runtime/env.py` (line 91)
  5. Update current environment with merged bindings (line 92)
- **Key limitation**: Path-insensitive join loses precision
  - Example: `if` branch assigns `A = zeros(3,4)`, `else` assigns `A = zeros(3,6)` → result is `matrix[3 x unknown]`
  - Test10.m demonstrates: `A + B` mismatch detected inside branch, but join loses branch-specific info

## Shape Domain (`runtime/shapes.py`)
- `Shape`: scalar | matrix[r x c] | unknown
- `Dim`: int | str (symbolic) | None (unknown)
- `join_dim(a, b)`: Returns `a` if `a==b`, else `None` (loses precision)
- `dims_definitely_conflict(a, b)`: Conservative check (returns False if either is None)

## Environment Operations (`runtime/env.py`)
- `Env`: Dict[str, Shape] with get/set/copy methods
- `join_env(env1, env2)`: Pointwise join using `join_shape` for each variable
- `join_shape(s1, s2)`: Joins shapes conservatively (scalar+scalar=scalar, matrix+matrix joins dims, otherwise unknown)

## Test Patterns
- Control flow tests: test10.m (if with mismatch), test11.m (suspicious comparison), test26.m (no control flow)
- Tests use symbolic dimensions extensively: `n`, `m`, `k`, `(k+m)`
- Warning codes use `W_*` prefix (stable)

## Spec Writing Best Practices
- **Scope narrowing critical**: Path-sensitive analysis is expensive; pragmatic subset needed
- **Test-first approach**: Define concrete test cases that demonstrate the improvement
- **Minimal invasiveness**: Changes should be localized to control flow handling
- **Backward compatibility**: Existing tests must continue passing
