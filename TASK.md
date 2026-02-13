# Task: Expand Builtin Whitelist and Fix Call Fallback Soundness

## Goal
Expand the parser's hardcoded builtin set to include common MATLAB functions and fix a soundness bug where unrecognized function calls return `scalar` instead of `unknown`.

## Scope
- `/root/projects/MATLAB_analysis/frontend/matlab_parser.py` line 348: Expand builtin set from `{"zeros", "ones", "size", "isscalar"}` to include `randn`, `eye`, `rand`, `sqrt`, `abs`, `length`, `numel`, `diag`, `inv`, `det`, `norm`, `linspace`, `reshape`, `transpose`, `repmat`
- `/root/projects/MATLAB_analysis/analysis/analysis_ir.py` line 152: Change `return Shape.scalar()` to `return Shape.unknown()`
- `/root/projects/MATLAB_analysis/analysis/diagnostics.py`: Add `warn_unknown_function()` returning `W_UNKNOWN_FUNCTION` diagnostic
- `/root/projects/MATLAB_analysis/tests/`: Add test28.m exercising expanded builtins and unknown function warning

## Non-goals
- Implementing precise shape rules for new builtins (deferred to Phase 2)
- Disambiguating existing indexing vs call ambiguities in test files
- Modifying the IR representation or lowering logic

## Invariants Impacted
- **IR analyzer is authoritative**: Preserved — changes are minimal and localized
- **Test expectations match IR analyzer**: Preserved — new test validates IR analyzer behavior
- **Conservative analysis soundness**: Strengthened — unknown calls now return `unknown` instead of incorrectly assuming `scalar`

## Acceptance Criteria
- [ ] Parser recognizes 19 builtins total (4 existing + 15 new) and generates Call nodes for them
- [ ] Unrecognized calls return `Shape.unknown()` instead of `Shape.scalar()`
- [ ] When an unbound variable is called, emit `W_UNKNOWN_FUNCTION` informational warning: "Function 'foo' is not recognized; treating result as unknown"
- [ ] New test demonstrates: (1) new builtins parse as calls, (2) unknown function emits warning, (3) result is unknown
- [ ] All existing tests pass: `python3 mmshape.py --tests`

## Commands to Run
```bash
# Run all tests
python3 /root/projects/MATLAB_analysis/mmshape.py --tests

# Run new test
python3 /root/projects/MATLAB_analysis/mmshape.py /root/projects/MATLAB_analysis/tests/test28.m

# Compare against legacy analyzer
python3 /root/projects/MATLAB_analysis/mmshape.py --compare /root/projects/MATLAB_analysis/tests/test28.m
```

## Tests to Add/Change

**Test file**: `/root/projects/MATLAB_analysis/tests/test28.m`
- Call one new builtin: `X = randn(3, 4);` — expects Call node parsed, result `unknown`
- Call unrecognized function: `Y = my_custom_func(5);` — expects `W_UNKNOWN_FUNCTION` warning, result `unknown`
- Use unknown result in computation: `Z = Y + 1;` — verifies `unknown` propagates correctly, not `scalar`
- Assertions:
  - `% EXPECT: warnings = 1`
  - `% EXPECT: X = unknown`
  - `% EXPECT: Y = unknown`
  - `% EXPECT: Z = unknown`
