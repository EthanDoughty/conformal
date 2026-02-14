# Documentation Maintainer Memory

## Common Drift Patterns

### Test Path References
- **Test organization**: Categorized subdirectories (basics/, symbolic/, indexing/, control_flow/, literals/, builtins/, recovery/)
- **Glob pattern**: tests/**/*.m (recursive) not tests/test*.m
- **Pattern**: When referencing tests, use descriptive category paths not numbered testN.m
- **Example paths**: tests/basics/valid_add.m, tests/recovery/struct_field.m, tests/builtins/constructors.m

### Version Documentation Flow
- **CHANGELOG.md**: Add new version entry with date
- **Format**: Keep a Changelog - sections: Added, Changed, Fixed, Removed
- **Detail Level**: Concise but specific (file names, warning codes, test numbers)
- **User Preference**: No Co-Authored-By lines in commits

## Project-Specific Conventions

### Warning Codes
- Prefix: `W_*` (e.g., W_UNKNOWN_FUNCTION, W_UNSUPPORTED_*)
- Document in: analysis/diagnostics.py (docstrings), CHANGELOG.md (when added)
- CLAUDE.md mentions warning code stability but does not list all codes

### Builtin Functions
- Source of truth: `KNOWN_BUILTINS` constant in analysis/builtins.py (centralized as of v0.8.6)
- Shape rules: `BUILTINS_WITH_SHAPE_RULES` in analysis/builtins.py
- **v0.8.7 milestone**: Full coverage (19/19 — KNOWN_BUILTINS == BUILTINS_WITH_SHAPE_RULES)
- CLAUDE.md and README.md do NOT list individual builtins (too volatile)
- Document new shape rules in CHANGELOG.md with function signatures
- New builtins added in Phase 3: det, diag, inv, linspace, norm

### Test Organization (Updated v0.8.5+)
- Discovery: Dynamic via `glob("tests/**/*.m", recursive=True)`
- Structure: Categorized subdirectories with descriptive names
- No more numbered test files - use category/descriptive_name.m format
- Example: tests/basics/valid_add.m not tests/test1.m

## Files to Always Check on Changes

### Parser/Frontend Changes
- CLAUDE.md: Architecture section (lines ~80-98)
- README.md: "What the Analysis Detects" (lines ~17-34)

### Test Suite Changes
- CLAUDE.md: Test file range (line ~134)
- README.md: Test count (line ~79) and test table (lines ~89-107)

### Warning/Diagnostic Changes
- CHANGELOG.md: Add entry for new warning codes
- analysis/diagnostics.py: Ensure docstrings mention warning codes

### Version Bumps
- CHANGELOG.md: Move items from Unreleased to new version section
- Use YYYY-MM-DD date format
- Keep Unreleased section for future changes
- Version format: `[0.8.2]` (no 'v' prefix in section headers)

### Symbolic Arithmetic Synchronization
- **CLAUDE.md line 122**: Lists symbolic arithmetic capabilities
- **README.md line 65**: Bulleted list of symbolic arithmetic features
- **Pattern**: When adding new symbolic operators (add_dim, mul_dim), update BOTH locations
- **v0.8.5 update**: Added multiplication support for repmat (mul_dim function)

## Recent Documentation Updates (v0.8.2-v0.8.5)

### v0.8.2 Architectural Change
- Unified Apply IR node for call-vs-index disambiguation
- Parser no longer distinguishes Call vs Index at parse time
- Analyzer disambiguates using colon heuristic and shape information
- Key insight: This is a major architectural shift (parse-time → analysis-time decision)
- Affects: README.md architecture mentions, test count (29→30)

### v0.8.3 Refactoring
- Extracted `_eval_indexing()` helper in analysis_ir.py
- Consolidated three copies of indexing logic
- Typed `Apply.args` properly as `List[IndexArg]`
- Pure refactor: no test expectation changes, no user-facing changes

### v0.8.4 Rich Builtin Shape Rules
- Added shape rules for 12 functions (up from ~4)
- Introduced `_eval_index_arg_to_shape()` helper
- Test31.m (now tests/builtins/constructors.m) with 38 assertions
- Fixed non-deterministic output in join_env() (sorted iteration)

### v0.8.5 Symbolic Multiplication
- Added `mul_dim()` function in runtime/shapes.py
- Shape rules for reshape and repmat
- Required updating symbolic arithmetic documentation in TWO places:
  - CLAUDE.md line 122 (Key features list)
  - README.md line 65 (analysis supports bullet list)
- New test: tests/builtins/reshape_repmat.m (7 assertions)
- Test count: 32 total

### v0.8.7 Complete Builtin Coverage (Phase 3)
- Shape rules for 5 new builtins: det, diag, inv, linspace, norm
- Fixed zeros(n)/ones(n) 1-arg gap (now returns n×n, was unknown)
- Achieved full coverage: BUILTINS_WITH_SHAPE_RULES == KNOWN_BUILTINS (19/19)
- New test file: tests/builtins/remaining_builtins.m (19 expectations: 1 warnings + 18 shapes)
- Test count: 44 total
- **Documentation impact**: CHANGELOG.md ✓, README.md ✓ (test table updated)

### v0.9.1 Dimension Arithmetic in Builtin Arguments
- Enhanced `expr_to_dim_ir` to handle BinOp (+, -, *) in dimension arguments
- Symbolic arithmetic now works in builtin calls: zeros(n+1, m), reshape(A, 2*n, m+1)
- Concrete arithmetic folding: zeros(2+3, 4*2) → matrix[5 x 8]
- New test file: tests/builtins/dim_arithmetic.m (9 cases, 0 warnings)
- Test count: 45 total
- **Documentation pattern**: Dimension capabilities appear in THREE places:
  1. CHANGELOG.md [Unreleased] — Feature description + test file
  2. CLAUDE.md line 122 — "Key features" symbolic arithmetic list
  3. README.md line 65 — "The analysis supports" bullet list
- **Key insight**: When symbolic dimension capabilities expand, check all three locations
