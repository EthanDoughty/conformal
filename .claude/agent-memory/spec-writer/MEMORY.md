# Spec Writer Memory

## Project Structure
- **IR analyzer is authoritative**: `analysis/analysis_ir.py` is the source of truth
- **Legacy analyzer**: `analysis/analysis_legacy.py` exists only for regression comparison
- **Test format**: Inline `% EXPECT:` assertions in .m files in `tests/` directory
- **Test discovery**: Dynamic via `glob("tests/**/*.m", recursive=True)` in `run_all_tests.py`
- **Test organization**: Categorized subdirectories (basics, symbolic, indexing, control_flow, literals, builtins, recovery)
- **Current test count**: 31 test files as of v0.8.4 (2026-02-13)

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
- `add_dim(a, b)`: Additive symbolic arithmetic (lines 93-108) — returns `a+b` for ints, `"(a+b)"` for symbolic, `None` if either is `None`
- `sum_dims(dimensions)`: Sums a list of dimensions using `add_dim` (lines 110-124)

## Environment Operations (`runtime/env.py`)
- `Env`: Dict[str, Shape] with get/set/copy methods
- `join_env(env1, env2)`: Pointwise join using `join_shape` for each variable
- `join_shape(s1, s2)`: Joins shapes conservatively (scalar+scalar=scalar, matrix+matrix joins dims, otherwise unknown)

## Test Patterns
- Control flow tests: tests/control_flow/if_branch_mismatch.m, tests/control_flow/suspicious_comparison.m
- Recovery tests: tests/recovery/*.m (struct_field.m, cell_array.m, multiple_assignment.m, multiline_braces.m, dot_elementwise.m, end_in_parens.m)
- Tests use symbolic dimensions extensively: `n`, `m`, `k`, `(k+m)`
- Warning codes use `W_*` prefix (stable)

## Parser and Call/Index Disambiguation (Phase 2 COMPLETE)
- Lines 9-14 of `frontend/matlab_parser.py`: `KNOWN_BUILTINS` set (19 functions as of v0.8.3)
- Unified `Apply` IR node implemented — parser no longer makes semantic decisions
- Analyzer (lines 165-224 of `analysis_ir.py`) disambiguates at analysis time based on:
  - Colon/Range in args → force indexing
  - Base is known builtin → function call
  - Base is unbound variable → emit `W_UNKNOWN_FUNCTION`, return unknown
  - Otherwise → treat as indexing

## Analyzer Call Handling (Phase 3 COMPLETE as of v0.8.4)
- Lines 201-285 of `analysis_ir.py`: Apply node evaluation with runtime disambiguation
- Line 16-23: `_BUILTINS_WITH_SHAPE_RULES` set tracks which builtins have implemented shape rules
- Currently handled (v0.8.4): `zeros`, `ones`, `eye`, `rand`, `randn`, `abs`, `sqrt`, `transpose`, `length`, `numel`, `size`, `isscalar`
- Still missing from 19-function whitelist: `reshape`, `repmat`, `linspace`, `diag`, `det`, `inv`, `norm`
- Other builtins in `KNOWN_BUILTINS` return `unknown` silently (line 276)
- Unrecognized functions emit `W_UNKNOWN_FUNCTION` and return `unknown` (lines 280-281)
- **Builtin shape rule patterns**:
  - Matrix constructors: Check arg count (0/1/2), use `expr_to_dim_ir` + `unwrap_arg`, return matrix shapes
  - Element-wise ops: Use `_eval_index_arg_to_shape` to get input shape, return same shape
  - Query functions: Evaluate arg to force side effects, return scalar
  - Transpose: Extract dimensions and swap them

## Warning Infrastructure (`analysis/diagnostics.py`)
- Warning functions return strings starting with `W_*` code
- Pattern: `def warn_X(line, ...) -> str:` returns formatted message
- `W_UNSUPPORTED_STMT` used for opaque statements
- All warnings include line number

## Spec Writing Best Practices
- **Scope narrowing critical**: Path-sensitive analysis is expensive; pragmatic subset needed
- **Test-first approach**: Define concrete test cases that demonstrate the improvement
- **Minimal invasiveness**: Changes should be localized to control flow handling
- **Backward compatibility**: Existing tests must continue passing
- **Read code before writing spec**: Ground specs in actual implementation, not assumptions
- **Non-goals are explicit**: List what task does NOT do to prevent scope creep
- **Builtin function patterns**: Check `KNOWN_BUILTINS` (parser.py:9-14) and `_BUILTINS_WITH_SHAPE_RULES` (analysis_ir.py:16)
- **Test current behavior first**: Run quick smoke test on `/tmp/` file to verify assumptions before speccing
- **Refactoring tasks need file migration maps**: For structural changes, provide old→new path mappings and concrete checklists
