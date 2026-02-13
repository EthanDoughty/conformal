# Implementer Memory

## Phase 1 Implementation (Expanded Builtins & Unknown Functions)
✅ COMPLETED — All 29 tests pass

### Key Files Modified
1. **frontend/matlab_parser.py** line 348: Builtin set expanded to 19 functions
   - Original: `{"zeros", "ones", "size", "isscalar"}`
   - Expanded: Added 15 more: `randn`, `eye`, `rand`, `sqrt`, `abs`, `length`, `numel`, `diag`, `inv`, `det`, `norm`, `linspace`, `reshape`, `transpose`, `repmat`
   - Only these recognized functions become Call nodes in IR

2. **analysis/diagnostics.py**: Added `warn_unknown_function(line, name)` function
   - Returns message with code `W_UNKNOWN_FUNCTION`
   - Used for truly unrecognized functions, NOT for builtins without shape rules

3. **analysis/analysis_ir.py** line 152+: Critical soundness fix
   - Changed fallback from `Shape.scalar()` to `Shape.unknown()` for unrecognized calls
   - Added silent handling for known builtins without shape rules (return `unknown` no warning)
   - Added warning emit for Index applied to unbound variables (e.g., `my_func(x)`)
   - Key insight: Index handles unbound variable calls since parser only creates Call nodes for recognized builtins

4. **tests/builtins/unknown_function.m**: Test for Phase 1 (renamed from test28.m)
   - Tests `randn` (recognized builtin, unknown result, silent)
   - Tests `my_custom_func` (unrecognized, emits warning, unknown result)
   - Tests propagation of `unknown` through arithmetic

## Phase 2 Implementation (Unified Apply Node)
✅ COMPLETED — All 30 tests pass (now 31 with Phase 3)

### Files Modified
[See detailed history above - Phase 2 completed]

## Phase 3 Implementation (Rich Builtin Shape Rules)
✅ COMPLETED — All 31 tests pass

### Files Modified
1. **analysis/analysis_ir.py** — Added shape rules for 10 builtins
   - Updated `_BUILTINS_WITH_SHAPE_RULES` set to include: eye, rand, randn, abs, sqrt, transpose, length, numel
   - Added `_eval_index_arg_to_shape()` helper — evaluates IndexArg to Shape (handles IndexExpr, Range, Colon)
   - Removed early-exit for Colon/Range (known builtins can accept ranges/colons)
   - Added shape rules in Apply handler:
     * **eye, rand, randn**: 0-arg→scalar, 1-arg→n×n, 2-arg→m×n (like zeros/ones)
     * **abs, sqrt**: pass-through shape (output = input shape)
     * **transpose**: swap dimensions (rows↔cols)
     * **length, numel**: return scalar

2. **tests/builtins/shape_preserving.m** — Tests shape-preserving builtins (renamed from test29.m)
   - Validates pass-through shape behavior

3. **tests/builtins/constructors.m** — Comprehensive builtin test (renamed from test31.m)
   - Tests all 10 builtin shape rules
   - Covers 0-arg, 1-arg, 2-arg forms
   - Tests concrete and symbolic dimensions
   - Tests edge case: `eye(0)` → `matrix[0 x 0]`
   - Tests Range arguments: `length(1:10)` → `scalar`

### Key Implementation Notes
- Changed architecture: Known builtins are checked BEFORE checking for Colon/Range
  - This allows builtins to accept Range/Colon args (e.g., `length(1:10)`)
  - Indexing forced only for unknown variables
- `_eval_index_arg_to_shape()` enables handling of Range/Colon in builtin args
  - Range → matrix[1 x None] (row vector)
  - Colon → unknown (cannot evaluate standalone)
- All 31 tests pass (Phase 1+2+3 complete)

### Architecture Decision
The original "Colon/Range force indexing" decision was too conservative for builtin functions.
Refined approach:
- Colon/Range force indexing for **unknown variables** (ambiguity → indexing)
- Colon/Range allowed in **known builtin** args (builtin can interpret them)
This maintains soundness while enabling valid patterns like `length(1:10)`.

## Test Suite Refactoring (Flat to Categorized)
✅ COMPLETED — All 31 tests pass in new structure

### Files Modified
1. **All 31 test files moved** using `git mv` (preserves history):
   - tests/test1-8.m → tests/basics/
   - tests/test5-6.m → tests/symbolic/
   - tests/test9,16-21.m → tests/indexing/
   - tests/test10-11.m → tests/control_flow/
   - tests/test13-15.m → tests/literals/
   - tests/test22-27.m → tests/recovery/
   - tests/test28-31.m → tests/builtins/

2. **run_all_tests.py** — Updated test discovery:
   - Line 17: Changed `test_sort_key()` from numeric suffix to alphabetical path
   - Line 23: Changed glob from `tests/test*.m` to `tests/**/*.m` with `recursive=True`
   - Sorting now fully alphabetical by path

3. **Documentation updated**:
   - CLAUDE.md: Updated example paths, glob pattern, strict mode recovery tests list
   - README.md: Updated test category descriptions with new paths
   - AGENTS.md: Updated glob patterns, example paths, recovery tests list
   - semantic-differential-auditor.md: Updated test reference format

### Key Verification
- All 31 tests still pass after move
- All 166 EXPECT lines preserved exactly
- New glob finds exactly 31 files
- `--strict` mode correctly detects recovery tests
- Alphabetical sorting is deterministic
