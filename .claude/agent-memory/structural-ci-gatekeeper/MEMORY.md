# Structural CI Gatekeeper Memory

## Test Infrastructure Patterns

### Test Discovery
- Pattern: glob("tests/test*.m") in run_all_tests.py:23
- Sorting: test_sort_key() extracts numeric suffix for deterministic ordering
- As of 2026-02-13: 31 test files (test1.m through test31.m)

### Expectation Format
- % EXPECT: warnings = N - warning count assertion
- % EXPECT: var = shape - shape assertion (scalar, matrix[r x c], unknown)
- Regex: EXPECT_RE, EXPECT_WARNINGS_RE, EXPECT_BINDING_RE in run_all_tests.py
- Shape normalization: whitespace stripped via normalize_shape_str()

### CLI Exit Codes
- 0: All tests passed OR single file analyzed successfully
- 1: Test failure OR parse error OR strict mode violation
- Strict mode: --strict flag fails if any W_UNSUPPORTED_* warnings present

### Warning Code Conventions
- All warning functions use warn_* naming (e.g., warn_matmul_mismatch)
- Two warnings use stable W_* prefix codes:
  - W_UNSUPPORTED_STMT - opaque/unsupported statements
  - W_UNKNOWN_FUNCTION - unrecognized function calls
- Most warnings use descriptive text without code prefix
- Strict mode check: has_unsupported() in diagnostics.py looks for W_UNSUPPORTED_* prefix

## Known Issues

### Non-Deterministic Output (CRITICAL)
**Status**: Confirmed structural bug in runtime/env.py:46-47

**Root cause**: join_env() uses unordered set iteration
**Fix**: Replace line 46 with sorted(set(...))
**Escalation**: Report to Ethan (Integrator)
