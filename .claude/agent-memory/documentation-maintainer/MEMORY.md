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
- Source of truth: `KNOWN_BUILTINS` constant in frontend/matlab_parser.py
- Currently 19 functions (as of v0.8.1)
- CLAUDE.md and README.md do NOT list individual builtins (too volatile)

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

## Recent Documentation Updates (v0.8.2-v0.8.3)

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
