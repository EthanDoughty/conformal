# Spec Writer Memory

## Project Structure
- **IR analyzer is authoritative**: `analysis/analysis_ir.py` is the source of truth
- **Legacy analyzer**: `analysis/analysis_legacy.py` exists only for regression comparison (--compare mode removed)
- **Test format**: Inline `% EXPECT:` assertions in .m files in `tests/` directory
- **Test discovery**: Dynamic via `glob("tests/**/*.m", recursive=True)` in `run_all_tests.py`
- **Test organization**: Categorized subdirectories (basics, symbolic, indexing, control_flow, literals, builtins, loops, recovery) — 8 dirs
- **Current test count**: 43 test files as of v0.8.7 (2026-02-13)

## Control Flow Current Implementation
- **If statements** (lines 113-126 of `analysis_ir.py`):
  1. Evaluate condition (line 114)
  2. Copy environment for then/else branches (lines 116-117)
  3. Analyze each branch independently (lines 119-122)
  4. Join environments using `join_env` from `runtime/env.py` (line 124)
  5. Update current environment with merged bindings (line 125)
  - Path-insensitive join loses precision (e.g., `if` assigns `matrix[3x4]`, `else` assigns `matrix[3x6]` → join is `matrix[3 x unknown]`)

- **While loops** (lines 100-103 of `analysis_ir.py`):
  - Evaluate condition once (line 101)
  - Analyze body via `_analyze_loop_body` (line 102) — supports fixpoint iteration
  - `--fixpoint` flag enables iterative convergence (lines 49-71)

- **For loops** (lines 105-111 of `analysis_ir.py`):
  - Binds loop variable to scalar (lines 106-107)
  - Evaluates iterator expression for side effects (line 109)
  - Analyze body via `_analyze_loop_body` (line 110) — supports fixpoint iteration
  - `--fixpoint` flag enables iterative convergence (MAX_LOOP_ITERATIONS = 3)

## Shape Domain (`runtime/shapes.py`)
- `Shape`: scalar | matrix[r x c] | unknown
- `Dim`: int | str (symbolic) | None (unknown)
- `join_dim(a, b)`: Returns `a` if `a==b`, else `None` (loses precision)
- `dims_definitely_conflict(a, b)`: Conservative check (returns False if either is None)
- `add_dim(a, b)`: Additive symbolic arithmetic (lines 93-108) — returns `a+b` for ints, `"(a+b)"` for symbolic, `None` if either is `None`
- `mul_dim(a, b)`: Multiplicative symbolic arithmetic (lines 110-137) — short-circuits 0 and 1, returns `a*b` for ints, `"(a*b)"` for symbolic
- `sum_dims(dimensions)`: Sums a list of dimensions using `add_dim` (lines 139-153)

## Environment Operations (`runtime/env.py`)
- `Env`: Dict[str, Shape] with get/set/copy methods (dataclass with `bindings: Dict[str, Shape]`)
- `join_env(env1, env2)`: Pointwise join using `join_shape` for each variable
- `join_shape(s1, s2)`: Joins shapes conservatively (scalar+scalar=scalar, matrix+matrix joins dims, otherwise unknown)
- **No `__eq__` method**: Env comparison requires `env1.bindings == env2.bindings` (works because Shape is frozen dataclass with `__eq__`)

## Test Patterns
- **Test count**: 43 test files as of v0.8.7 (2026-02-13)
- Control flow tests: tests/control_flow/if_branch_mismatch.m, tests/control_flow/suspicious_comparison.m
- Recovery tests: tests/recovery/*.m (struct_field.m, cell_array.m, multiple_assignment.m, multiline_braces.m, dot_elementwise.m, end_in_parens.m)
- Builtin tests: tests/builtins/constructors.m (comprehensive coverage of eye/rand/randn/abs/sqrt/transpose/length/numel), tests/builtins/reshape_repmat.m, tests/builtins/shape_preserving.m
- Loop tests: tests/loops/*.m (8 test files covering fixpoint iteration)
- Tests use symbolic dimensions extensively: `n`, `m`, `k`, `(k+m)`, `(n*k)`
- Warning codes use `W_*` prefix (stable)

## Parser and Call/Index Disambiguation (Phase 2 COMPLETE)
- `analysis/builtins.py`: Contains `KNOWN_BUILTINS` set (19 functions) and `BUILTINS_WITH_SHAPE_RULES` set
- Unified `Apply` IR node implemented — parser no longer makes semantic decisions
- Analyzer (lines 193-317 of `analysis_ir.py`) disambiguates at analysis time based on:
  - Colon/Range in args → force indexing
  - Base is known builtin → function call
  - Base is unbound variable → emit `W_UNKNOWN_FUNCTION`, return unknown
  - Otherwise → treat as indexing

## Analyzer Call Handling (v0.8.7 as of 2026-02-13)
- Lines 193-317 of `analysis_ir.py`: Apply node evaluation with runtime disambiguation
- `analysis/builtins.py` (lines 20-28): `BUILTINS_WITH_SHAPE_RULES` set tracks which builtins have implemented shape rules
- Handled (v0.8.7): `zeros`, `ones`, `eye`, `rand`, `randn`, `abs`, `sqrt`, `transpose`, `length`, `numel`, `size`, `isscalar`, `reshape`, `repmat`
- Missing from 19-function whitelist: `linspace`, `diag`, `det`, `inv`, `norm` (5 builtins need shape rules — **Phase 3 task**)
- Other builtins in `KNOWN_BUILTINS` return `unknown` silently (line 308)
- Unrecognized functions emit `W_UNKNOWN_FUNCTION` and return `unknown` (lines 310-313)
- **Builtin shape rule patterns**:
  - Matrix constructors: Check arg count (0/1/2), use `expr_to_dim_ir` + `unwrap_arg`, return matrix shapes
  - Element-wise ops: Use `_eval_index_arg_to_shape` to get input shape, return same shape
  - Query functions: Evaluate arg to force side effects, return scalar
  - Transpose: Extract dimensions and swap them
  - reshape: Takes 3 args (matrix, m, n), returns matrix[m x n]
  - repmat: Takes 3 args (matrix, m, n), returns matrix[a_rows*m x a_cols*n] using `mul_dim`
  - All handlers use `try/except ValueError` to catch colon/range in args

## Warning Infrastructure (`analysis/diagnostics.py`)
- Warning functions return strings starting with `W_*` code
- Pattern: `def warn_X(line, ...) -> str:` returns formatted message
- `W_UNSUPPORTED_STMT` used for opaque statements
- All warnings include line number

## Dimension Extraction (`expr_to_dim_ir`)
- **Current implementation**: Lines 507-524 of `analysis_ir.py`
- **Supported**: `Const` → int, `Var` → symbolic name, everything else → `None`
- **Missing**: BinOp handling (arithmetic on dimensions)
- **Usage sites**: All builtin constructors that take dimension args (`zeros`, `ones`, `eye`, `rand`, `randn`, `reshape`, `repmat`, `linspace`)
- **Extension pattern**: Recurse into `BinOp` nodes, use `add_dim`/`mul_dim` from shapes.py
- **Current precision loss**: `zeros(n+1, m)` → `matrix[None x m]` instead of `matrix[(n+1) x m]`

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
- **Check existing implementation first**: Before writing a spec, verify the feature isn't already implemented (e.g., Phase 2 Apply node was already done)
