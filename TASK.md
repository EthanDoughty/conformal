# Task: Polymorphic Caching and Return Statement Support (v0.10.1)

## Goal
Optimize user-defined function analysis by caching results per unique argument shape tuple. Add explicit `return` statement support with early exit semantics via `EarlyReturn` exception with catch-at-boundary precision in control flow.

## Scope
- **Polymorphic caching**: Cache `(List[Shape], List[str])` (output shapes + raw warnings) keyed by `(func_name, tuple of arg shapes)`
- **Warning replay**: On cache hit, replay cached warnings with current call site's dual-location formatting
- **No custom `Shape.__hash__`/`__eq__`**: `@dataclass(frozen=True)` already provides correct implementations
- **Explicit return**: Parse `return` keyword, add `Return` IR node, raise `EarlyReturn` exception
- **Catch-at-boundary**: Catch `EarlyReturn` in `If` (join non-returned branch) and loop handlers (stop iteration, don't propagate)
- **Return in script**: Stop analysis of subsequent statements (raise `EarlyReturn`, catch in `analyze_program_ir`)
- 5 new tests (2 caching, 1 cache+warning, 2 return)

## Non-goals
- Cross-file cache persistence
- Cache eviction/LRU policy
- Custom `Shape.__hash__`/`__eq__` (frozen dataclass already provides)
- Symbolic expression canonicalization (n+m != m+n for cache keys)
- Multiple return points with different output shapes per branch

## Invariants Impacted
- **IR analyzer authoritative**: Preserved
- **All tests pass**: 66 existing + 5 new = 71
- **Deterministic**: Cache is deterministic
- **`analyze_stmt_ir` signature unchanged**: Returns `Env`, early exit via exception

## Acceptance Criteria
- [ ] `AnalysisContext.analysis_cache: Dict[Tuple[str, Tuple[Shape, ...]], Tuple[List[Shape], List[str]]]`
- [ ] Cache hit replays warnings with dual-location formatting for current call site
- [ ] Cache miss analyzes and stores `(result_shapes, raw_warnings)` before returning
- [ ] Parser recognizes `return` keyword
- [ ] `Return` IR node (no expression — MATLAB return has no value)
- [ ] `EarlyReturn` exception raised by `Return` handler in function context
- [ ] `EarlyReturn` caught at boundaries: `If` handler, `_analyze_loop_body`, `analyze_function_call`, `analyze_program_ir`
- [ ] `If` handler: if one branch returns, result env is the non-returned branch's env; if both return, re-raise
- [ ] Loop handler: catch `EarlyReturn`, do post-loop join, don't propagate
- [ ] Script-level `return`: emit `W_RETURN_OUTSIDE_FUNCTION`, stop analysis (raise `EarlyReturn`, catch in `analyze_program_ir`)
- [ ] All 71 tests pass (66 existing + 5 new)

## Commands to Run
```bash
python3 mmshape.py --tests
python3 mmshape.py --fixpoint --tests
python3 mmshape.py tests/functions/cache_hit.m
python3 mmshape.py tests/functions/cache_miss.m
python3 mmshape.py tests/functions/cache_hit_with_warning.m
python3 mmshape.py tests/functions/return_statement.m
python3 mmshape.py tests/functions/return_in_script.m
```

## Tests to Add

### `tests/functions/cache_hit.m`
```matlab
% Test: Polymorphic cache hit (same argument shapes)
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT: C = matrix[3 x 3]

function y = square_matrix(x)
    y = x * x;
end

A = zeros(3, 3);
B = square_matrix(A);
C = square_matrix(zeros(3, 3));
```

### `tests/functions/cache_miss.m`
```matlab
% Test: Polymorphic cache miss (different argument shapes)
% EXPECT: warnings = 1
% EXPECT: B = unknown
% EXPECT: D = matrix[5 x 5]

function y = square_it(x)
    y = x * x;
end

A = zeros(3, 4);
B = square_it(A);
C = zeros(5, 5);
D = square_it(C);
```

### `tests/functions/cache_hit_with_warning.m`
```matlab
% Test: Cache hit must replay warnings at each call site
% EXPECT: warnings = 2
% EXPECT: B = unknown
% EXPECT: C = unknown

function y = bad_multiply(x)
    y = x * x;
end

A = zeros(3, 4);
B = bad_multiply(A);
C = bad_multiply(A);
```

### `tests/functions/return_statement.m`
```matlab
% Test: Return statement (early exit from function)
% EXPECT: warnings = 0
% EXPECT: A = scalar

function result = early_exit(x)
    result = x;
    return;
end

A = early_exit(5);
```

### `tests/functions/return_in_script.m`
```matlab
% Test: Return in script context stops analysis
% EXPECT: warnings = 1

A = zeros(3, 3);
return;
B = A * A;
```

## Design Notes

### No Custom Shape Hashing
`@dataclass(frozen=True)` auto-generates `__hash__` and `__eq__` based on all fields (`kind`, `rows`, `cols`). These are correct for cache keys — structural equality matches semantic equality.

### Cache Structure
```python
@dataclass
class AnalysisContext:
    function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)
    analyzing_functions: Set[str] = field(default_factory=set)
    analysis_cache: Dict[Tuple[str, Tuple[Shape, ...]], Tuple[List[Shape], List[str]]] = field(default_factory=dict)
    fixpoint: bool = False
```

Cache value is `(output_shapes, raw_warnings)` where raw_warnings are the function-internal warnings BEFORE dual-location formatting.

### Cache Logic in analyze_function_call
```python
# Evaluate arg shapes for cache key
arg_shapes = tuple(_eval_index_arg_to_shape(arg, env, warnings, ctx) for arg in args)
cache_key = (func_name, arg_shapes)

# Check cache
if cache_key in ctx.analysis_cache:
    cached_shapes, cached_warnings = ctx.analysis_cache[cache_key]
    # Replay warnings with current call site's dual-location formatting
    for func_warn in cached_warnings:
        formatted = _format_dual_location_warning(func_warn, func_name, line)
        warnings.append(formatted)
    return list(cached_shapes)  # Return copy

# ... analyze function body ...
# Store raw warnings (before dual-location formatting)
ctx.analysis_cache[cache_key] = (result, list(func_warnings))
# Then format warnings for current call site
for func_warn in func_warnings:
    formatted = _format_dual_location_warning(func_warn, func_name, line)
    warnings.append(formatted)
```

Extract dual-location formatting into `_format_dual_location_warning(warn, func_name, call_line)` helper.

### EarlyReturn Exception
```python
class EarlyReturn(Exception):
    """Raised by return statement to exit function body analysis."""
    pass
```

### Return Handler in analyze_stmt_ir
```python
if isinstance(stmt, Return):
    if not ctx.analyzing_functions:
        # Script context: warn and stop
        warnings.append(diag.warn_return_outside_function(stmt.line))
    raise EarlyReturn()
```

### Catch-at-Boundary: If Handler
```python
if isinstance(stmt, If):
    _ = eval_expr_ir(stmt.cond, env, warnings, ctx)
    then_env = env.copy()
    else_env = env.copy()

    then_returned = False
    else_returned = False

    try:
        for s in stmt.then_body:
            analyze_stmt_ir(s, then_env, warnings, ctx)
    except EarlyReturn:
        then_returned = True

    try:
        for s in stmt.else_body:
            analyze_stmt_ir(s, else_env, warnings, ctx)
    except EarlyReturn:
        else_returned = True

    if then_returned and else_returned:
        # Both branches return — propagate
        raise EarlyReturn()
    elif then_returned:
        # Only then returns — use else env
        env.bindings = else_env.bindings
    elif else_returned:
        # Only else returns — use then env
        env.bindings = then_env.bindings
    else:
        # Neither returns — normal join
        merged = join_env(then_env, else_env)
        env.bindings = merged.bindings
    return env
```

### Catch-at-Boundary: Loop Handler
```python
def _analyze_loop_body(body, env, warnings, ctx):
    if not ctx.fixpoint:
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except EarlyReturn:
            pass  # Stop iteration, don't propagate
        return

    # Fixpoint mode: catch EarlyReturn in each phase
    pre_loop_env = env.copy()
    try:
        for s in body:
            analyze_stmt_ir(s, env, warnings, ctx)
    except EarlyReturn:
        pass  # Phase 1 stopped early

    widened = widen_env(pre_loop_env, env)
    if widened.bindings != env.bindings:
        env.bindings = widened.bindings
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except EarlyReturn:
            pass  # Phase 2 stopped early

    final = widen_env(pre_loop_env, env)
    env.bindings = final.bindings
```

### Catch in analyze_function_call
```python
try:
    for stmt in sig.body:
        analyze_stmt_ir(stmt, func_env, func_warnings, ctx)
except EarlyReturn:
    pass  # Function returned early — outputs are current env values
```

### Catch in analyze_program_ir
```python
try:
    for item in program.body:
        if not isinstance(item, FunctionDef):
            analyze_stmt_ir(item, env, warnings, ctx)
except EarlyReturn:
    pass  # Script-level return stops analysis
```

### Parser Changes
Add `"return"` to KEYWORDS. In `parse_stmt()`, check for RETURN token before falling through to `parse_simple_stmt()`.

### Diagnostics
```python
def warn_return_outside_function(line: int) -> str:
    """Warning for return statement outside function body."""
    return f"W_RETURN_OUTSIDE_FUNCTION line {line}: return statement outside function body"
```

## Estimated Diff
- Parser: +10 lines (return keyword + parse_return)
- IR: +8 lines (Return node)
- Lowering: +4 lines (return case)
- Analyzer: +80 lines (cache logic, EarlyReturn, catch-at-boundary, warning helper)
- Diagnostics: +4 lines (warn_return_outside_function)
- Tests: 5 new files (~55 lines)
- **Total: ~161 lines** across 6 files + 5 test files
