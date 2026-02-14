# Task: Test Coverage Polish Pass (v0.10.2)

## Goal
Fill identified test coverage gaps in function features (v0.10.0-v0.10.1) by adding targeted tests for return statement edge cases, cache interactions, and function+control flow combinations.

## Scope
- Add 9 new test files covering undertested code paths
- Extend 2 existing test files with additional scenarios
- Focus on: return in if/loop, cache with symbolic args, multi-return + early return, function calls inside loops
- All new tests in `tests/functions/` directory

## Non-goals
- No analyzer code changes (pure test addition)
- No new features or shape rules
- No refactoring of existing tests
- No changes to test infrastructure or runner

## Invariants Impacted
- **All tests pass**: 71 existing + 11 new = 82 total
- **IR analyzer authoritative**: Preserved (no code changes)
- **Test format**: Standard `% EXPECT:` inline assertions

## Acceptance Criteria
- [ ] All 82 tests pass: `python3 mmshape.py --tests`
- [ ] All 82 tests pass in fixpoint mode: `python3 mmshape.py --fixpoint --tests`
- [ ] New tests cover identified gaps: return in if-branch, return in loop, cache+symbolic, cache+warning replay, multi-return+early exit, function in loop, procedure+return
- [ ] Test files follow existing naming conventions and organization
- [ ] Each test has clear description comment explaining what it validates

## Commands to Run
```bash
python3 mmshape.py --tests
python3 mmshape.py --fixpoint --tests
python3 mmshape.py tests/functions/return_in_if.m
python3 mmshape.py tests/functions/return_in_loop.m
python3 mmshape.py tests/functions/cache_symbolic_args.m
python3 mmshape.py tests/functions/cache_warning_replay.m
python3 mmshape.py tests/functions/early_return_multi_output.m
python3 mmshape.py tests/functions/function_in_loop.m
python3 mmshape.py tests/functions/nested_function_calls.m
python3 mmshape.py tests/functions/procedure_with_return.m
python3 mmshape.py tests/functions/arg_count_mismatch_cached.m
```

## Tests to Add

### 1. `tests/functions/return_in_if.m` (NEW)
**Gap**: No test for return inside if-branch in function body (auditor finding #1)
**Code path**: `If` handler catches `EarlyReturn` from then-branch, uses else-branch env
```matlab
% Test: Return statement inside if-branch
% If then-branch returns early, result env is else-branch env
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

function y = conditional_return(x, cond)
    if cond
        y = x;
        return;
    else
        y = x + x;
    end
end

A = conditional_return(zeros(4, 4), 1);
```

### 2. `tests/functions/return_in_loop.m` (NEW)
**Gap**: No test for return inside loop body in function (auditor finding #2)
**Code path**: `_analyze_loop_body` catches `EarlyReturn` at boundary, doesn't propagate
```matlab
% Test: Return statement inside loop body
% Loop catches EarlyReturn, does post-loop join, doesn't propagate to caller
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]

function y = loop_with_return(x)
    for i = 1:10
        if i > 5
            y = x;
            return;
        end
    end
    y = zeros(1, 1);
end

A = loop_with_return(zeros(3, 3));
```

### 3. `tests/functions/cache_symbolic_args.m` (NEW)
**Gap**: Cache with symbolic dimension arguments not explicitly tested (auditor finding #3)
**Code path**: Cache key includes symbolic shapes, hits when symbolic names match
```matlab
% Test: Polymorphic cache with symbolic dimension arguments
% Cache key is (func_name, (arg_shapes...)), symbolic dims must match exactly
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x n]
% EXPECT: B = matrix[n x n]
% EXPECT: C = matrix[m x m]

function y = make_square(x)
    y = x * x;
end

A = make_square(zeros(n, n));
B = make_square(zeros(n, n));  % Cache hit (same symbolic n)
C = make_square(zeros(m, m));  % Cache miss (different symbolic m)
```

### 4. `tests/functions/cache_warning_replay.m` (NEW)
**Gap**: Cache hit warning replay with dual-location formatting not tested separately
**Code path**: Cache hit replays warnings via `_format_dual_location_warning`
```matlab
% Test: Cache hit replays warnings at each call site with correct line numbers
% First call analyzes and caches warning; second call replays at different line
% EXPECT: warnings = 2
% EXPECT: A = unknown
% EXPECT: B = unknown

function y = inner_mismatch(x)
    y = x * x;
end

A = inner_mismatch(zeros(3, 4));  % Line 10: cache miss, emit warning
B = inner_mismatch(zeros(3, 4));  % Line 11: cache hit, replay warning
```

### 5. `tests/functions/early_return_multi_output.m` (NEW)
**Gap**: Early return in multi-output function not tested
**Code path**: Multiple output vars, return early, unset outputs become unknown
```matlab
% Test: Early return in multi-output function leaves some outputs unset
% output2 never assigned before return → bottom → unknown at boundary
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = unknown

function [output1, output2] = partial_return(x)
    output1 = x;
    return;
    output2 = x';
end

[A, B] = partial_return(zeros(3, 3));
```

### 6. `tests/functions/function_in_loop.m` (NEW)
**Gap**: Function call inside loop body not tested (potential cache interaction)
**Code path**: Loop calls function multiple times, cache should hit on 2nd+ iterations
```matlab
% Test: Function called inside loop body (cache interaction)
% Each iteration calls func with same arg shape → cache hits after first
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]

function y = identity(x)
    y = x;
end

A = zeros(3, 3);
for i = 1:5
    A = identity(A);
end
```

### 7. `tests/functions/nested_function_calls.m` (NEW)
**Gap**: Function calling another user-defined function not tested
**Code path**: Recursive registry lookup, nested analyzing_functions tracking
```matlab
% Test: Nested user-defined function calls (no recursion)
% outer calls inner; both should analyze correctly
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

function y = inner(x)
    y = x + x;
end

function z = outer(w)
    z = inner(w);
end

A = outer(zeros(4, 4));
```

### 8. `tests/functions/procedure_with_return.m` (NEW)
**Gap**: Procedure (no outputs) with explicit return not tested
**Code path**: Return in procedure, no output vars to extract
```matlab
% Test: Procedure with explicit return statement
% No output vars, return just exits early
% EXPECT: warnings = 1
% EXPECT: A = unknown

function myproc(x)
    y = x * x;
    return;
end

A = myproc(zeros(3, 3));
```

### 9. `tests/functions/arg_count_mismatch_cached.m` (NEW)
**Gap**: Argument count mismatch error path not tested (cache shouldn't store bad calls)
**Code path**: `len(args) != len(sig.params)` early-exits before cache check
```matlab
% Test: Function called with wrong number of arguments
% Arg count mismatch → warning, return unknown, no cache interaction
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = two_args(x, z)
    y = x + z;
end

A = two_args(zeros(3, 3));  % Missing 2nd arg
```

### 10. EXTEND `tests/functions/return_statement.m`
**Gap**: Current test only has return after assignment; add case with unreachable code
**Addition**: Add lines to demonstrate statements after return are not analyzed
```matlab
% Test: Return statement (early exit from function)
% EXPECT: warnings = 0
% EXPECT: A = scalar
% EXPECT: B = scalar

function result = early_exit(x)
    result = x;
    return;
end

function result2 = unreachable_after_return(x)
    result2 = x;
    return;
    result2 = zeros(100, 100);  % Not analyzed (dead code after return)
end

A = early_exit(5);
B = unreachable_after_return(10);
```

### 11. EXTEND `tests/functions/multiple_returns.m`
**Gap**: Current test uses both outputs; add case where caller ignores one output
**Addition**: Test single-output destructuring of multi-output function
```matlab
% Test: Function with multiple return values
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[4 x 3]
% EXPECT: C = matrix[3 x 4]

function [out1, out2] = transpose_pair(in1)
    out1 = in1;
    out2 = in1';
end

[A, B] = transpose_pair(zeros(3, 4));
C = transpose_pair(zeros(3, 4));  % Only use first output (implicitly)
```

## Coverage Gaps Addressed

### Auditor-Identified Gaps (Resolved)
1. **Return in if-branch**: `return_in_if.m` tests `EarlyReturn` caught by `If` handler
2. **Return in loop**: `return_in_loop.m` tests `EarlyReturn` caught by loop handler
3. **Cache + symbolic dims**: `cache_symbolic_args.m` tests cache keys with symbolic shapes

### Additional Gaps Found
4. **Cache warning replay**: `cache_warning_replay.m` tests dual-location warning formatting
5. **Early return + multi-output**: `early_return_multi_output.m` tests unset outputs → unknown
6. **Function in loop**: `function_in_loop.m` tests cache interaction with loop iterations
7. **Nested function calls**: `nested_function_calls.m` tests user-function calling user-function
8. **Procedure + return**: `procedure_with_return.m` tests return in zero-output function
9. **Arg count mismatch**: `arg_count_mismatch_cached.m` tests error before cache lookup
10. **Unreachable code after return**: Extended `return_statement.m` tests dead code not analyzed
11. **Multi-output single-use**: Extended `multiple_returns.m` tests caller using only first output

## Estimated Diff
- **9 new test files**: ~110 lines total (avg 12 lines per test)
- **2 extended test files**: ~20 lines of additions
- **Total**: ~130 lines of new test code, 0 lines of analyzer changes
- **Test count**: 71 → 82 (+11 tests)
