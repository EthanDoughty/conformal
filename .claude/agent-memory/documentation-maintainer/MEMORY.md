# Documentation Maintainer Memory

## Common Drift Patterns

### Test Range References
- **CLAUDE.md line ~134**: Test file range (currently "test1.m through test29.m")
- **README.md line ~79**: Total test count (currently 29 programs)
- **README.md lines ~89-107**: Test category table (maps test numbers to categories)
- **Pattern**: When new tests are added, all three locations must be updated

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

### Test Numbering
- Discovery: Dynamic via `glob("tests/test*.m")`
- Renumbering: When inserting tests mid-sequence, renumber subsequent tests
- Example: v0.8.1 added test28.m, renumbered old test28.m to test29.m

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
