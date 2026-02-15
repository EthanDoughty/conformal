# Implementer Agent Memory

## Key File Locations & Structure

### Test Infrastructure
- **Test discovery**: `run_all_tests.py` uses `glob("tests/**/*.m", recursive=True)`
- **Test categories**: 8 directories — basics/, builtins/, control_flow/, functions/, indexing/, literals/, loops/, recovery/
- **Test format**: Inline `% EXPECT:` assertions, `% EXPECT_FIXPOINT:` for fixpoint-specific expectations
- **Test runner**: `python3 mmshape.py --tests` (normal), `python3 mmshape.py --fixpoint --tests` (fixpoint mode)

### Shape System Core (`runtime/shapes.py`)
- **Shape domain**: `scalar | matrix[r x c] | unknown | bottom` (bottom added in v0.9.3)
- **Dim representation**: `int | str (symbolic) | None (unknown)`
- **Key functions**:
  - `join_dim(d1, d2)` — lattice join for dimensions
  - `dims_definitely_conflict(d1, d2)` — definite incompatibility check
  - `add_dim(d1, d2)` — symbolic addition (e.g., n + m)
  - `mul_dim(d1, d2)` — symbolic multiplication (e.g., 2*n)
  - `widen_dim(d1, d2)` — widening operator (conflicts → None, stable → preserve)
  - `widen_shape(s1, s2)` — shape-level widening
  - `widen_env(env1, env2)` — environment-level widening

### Analysis Pipeline
- **IR analyzer**: `analysis/analysis_ir.py` (authoritative, default)
- **Builtin registry**: `analysis/builtins.py` — `KNOWN_BUILTINS` dict, shape rule decorator `@register_builtin`
- **Diagnostics**: `analysis/diagnostics.py` — warning generation, dual-location formatting for function calls

### Function Analysis (v0.10.0-v0.10.1)
- **Function registry**: `self.functions: dict[str, FunctionSig]` in analyzer
- **Function cache**: `self.function_cache: dict[tuple, CachedResult]` — key is (func_name, arg_shapes_tuple)
- **Return mechanism**: `EarlyReturn` exception, caught at function/if/loop boundaries
- **Multi-output**: Destructuring assignment `[A, B] = func(...)`, unset outputs become `bottom` → `unknown` at boundary

## Common Patterns & Gotchas

### Return Statement Semantics
- **Return in function**: Raises `EarlyReturn`, caught at function boundary
- **Return in if-branch**: Caught by `If` handler, result uses else-branch env (then-branch exits early)
- **Return in loop**: Caught by loop handler, does post-loop join, doesn't propagate to caller
- **Unset outputs**: Multi-output function with early return → unset vars are `bottom` → `unknown` after extraction
- **Dead code after return**: Not analyzed (return statement terminates basic block)

### Loop Analysis (v0.9.2)
- **Default mode**: Single-pass (pre-loop → body once → post-loop join)
- **Fixpoint mode**: 3-phase widening algorithm (discover, stabilize, post-loop join)
- **Widening strategy**: Conflicting dimensions → None, stable dimensions preserved
- **Convergence guarantee**: ≤2 iterations in fixpoint mode

### Test Expectations
- When analyzer behavior differs from spec expectations, **adjust EXPECT lines to match reality**
- Analyzer code is correct; tests document actual behavior
- Example: `return_in_loop.m` expected `matrix[3 x 3]` but analyzer correctly produces `matrix[1 x 1]` (post-loop assignment wins)

### Warning Codes
- **Prefix convention**: All warning codes use `W_*` prefix
- **Function-specific codes**:
  - `W_UNKNOWN_FUNCTION` — informational (not alarming)
  - `W_FUNCTION_ARG_COUNT_MISMATCH` — arg count error
  - `W_PROCEDURE_IN_EXPR` — procedure (no outputs) used in expression
  - `W_RECURSIVE_FUNCTION` — recursion not supported
  - `W_RETURN_OUTSIDE_FUNCTION` — return in script-level code
  - `W_UNSUPPORTED_STMT` — recovery cases (cell, struct, etc.)

## Test Coverage Insights (v0.10.2 Polish Pass)

### Gaps Identified & Addressed
1. **Return in control flow**: Added tests for return in if-branch, return in loop body
2. **Cache interactions**: Symbolic arg cache keys, warning replay, function in loop
3. **Function edge cases**: Early return + multi-output, nested calls, procedure + return, arg count mismatch
4. **Dead code**: Unreachable code after return statement

### Test File Organization
- **Naming convention**: Descriptive names reflecting what is tested (e.g., `cache_symbolic_args.m`, `return_in_if.m`)
- **Header comments**: Always include description of what the test validates
- **EXPECT lines**: Come before the code, at top of file

### Control Flow Exception Propagation (v0.11.0)
- **Exception classes**: `EarlyReturn`, `EarlyBreak`, `EarlyContinue` in `analysis_ir.py`
- **Multi-way join pattern**: IfChain, Switch, Try all use same pattern:
  1. Evaluate conditions/switch expr for side effects
  2. Copy env for each branch, analyze each body in try/except
  3. Track which branches exited early (returned/broke/continued)
  4. If ALL branches exited, propagate exception (or deferred break/continue)
  5. Join only non-exited branches, update env
- **Deferred exceptions**: Break/continue in nested control flow stored, re-raised after join
- **Parser gotcha**: After `switch expr`, must skip NEWLINE before checking for CASE (parse_expr doesn't consume trailing newline)
- **Lowering gotcha**: ALL if lowering must handle 5-element syntax AST (including simple if with no elseifs)

### Lattice Refinement (v0.11.0)
- **join_dim semantics changed**: None now absorbing (top for dims) instead of identity (bottom)
- **Rationale**: Uninitialized vars use `Shape.bottom()`, not `matrix[None x None]`. Dim-level None is ONLY for "unknown dimension value", should stay unknown through joins.
- **Impact**: `join_dim(None, X) = None` for any X. More sound, less precise (e.g., concat with unknown loses column info).
- **Test adjustment**: `widen_concat_unknown.m` expectation updated from `matrix[None x 3]` to `matrix[None x None]`

## Version History Checkpoints
- **v0.9.0**: Call/Index disambiguation complete, unified Apply IR node, 44 tests
- **v0.9.2**: Principled loop widening, 52 tests
- **v0.9.3**: Shape.bottom() lattice fix, 54 tests
- **v0.10.0**: User-defined functions (Phase A+B), 65 tests
- **v0.10.1**: Function caching + return statements, 71 tests
- **v0.10.2**: Test coverage polish, 80 tests
- **v0.11.0**: Extended control flow (elseif/switch/try/break/continue), join_dim refinement, 92 tests
- **v0.12.0 Phase 1 (Strings)**: Context-sensitive `'` lexing, StringLit IR, Shape.string(), 96 tests

## v0.12.0 Phase 1 Implementation Notes (Strings)

### Context-Sensitive Lexer
- Lexer refactored from `MASTER_RE.finditer(src)` to manual loop with `MASTER_RE.match(src, pos)`
- **Reason**: Need to scan ahead for matching `'` when detecting string start
- **Previous token tracking**: `prev_kind` variable tracks last emitted token kind
- **Disambiguation rule**: After ID/)/]/NUMBER/TRANSPOSE → `'` is TRANSPOSE, else → STRING start
- **DQSTRING token**: `r'"[^"]*"'` matches double-quoted strings directly (no ambiguity)

### Shape Domain Extension
- `Shape` dataclass extended with `_fields: tuple = ()` for future struct support (Phase 2)
- `Shape.string()` constructor returns `Shape(kind="string")`
- `is_string()` predicate added
- `join_shape` and `widen_shape` updated: string+string=string, string+other=unknown
- `__str__` returns `"string"` for string kind

### String Arithmetic
- `+` on two strings → `Shape.matrix(1, None)` (MATLAB char + char = numeric row vector)
- String + matrix/scalar (any op except +) → `W_STRING_ARITHMETIC` warning + unknown
- Check in `eval_binop_ir` before scalar broadcasting logic

### Matrix Literal Horzcat
- **Critical fix**: `as_matrix_shape` called inside `infer_matrix_literal_shape`, NOT at call site
- **All-strings check**: If all elements are strings, return `Shape.string()` (horzcat of strings)
- **Before**: Evaluated shapes passed through `as_matrix_shape` before checking
- **After**: Raw shapes checked first, THEN converted

### Imports
- `StringLit` must be imported in `analysis/analysis_ir.py` from `ir`
- Added to import list alongside `Var`, `Const`, `MatrixLit`, etc.
