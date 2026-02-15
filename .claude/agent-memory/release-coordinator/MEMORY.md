# Release Coordinator Memory

## User Preferences
- Single-sentence commit messages (no Co-Authored-By line)
- Decimal versioning (0.8.6, NOT v0.8.6)
- Always ask approval before committing

## Release Patterns
- Minor bump: New features, backward compatible
- Patch bump: Bug fixes only
- Major bump: Breaking changes

## This Project's Version History
- 0.9.1: Dimension arithmetic in builtin arguments (expr_to_dim_ir handles BinOp)
- 0.9.0: Complete builtin shape rules (19/19 coverage)
- 0.8.6: Fixed-point iteration for loop analysis
- 0.8.5: reshape/repmat shape rules
- 0.8.4: Rich builtin shape rules (12 functions)
- 0.8.3: Extracted indexing helper
- 0.8.2: Unified Apply IR node
- 0.8.1: Expanded builtins to 19 functions

## Release Decision Criteria
- All tests pass (45/45 in both modes as of 0.9.1)
- Quality score not tracked (no QA agent ran)
- Documentation synchronized (user confirmed)
- Breaking changes: Requires major bump
- New feature + bug fix: Minor bump (precedent: 0.8.4 added features + fixed non-determinism)

## Commit Message Patterns
- Format: "Brief description (version)" or "Brief description, additional context"
- Examples from recent releases:
  - "Add fixed-point iteration for loop analysis with --fixpoint flag (0.8.6)"
  - "Add dimension arithmetic in builtin arguments (0.9.1)"
  - "Complete builtin shape rules for all 19 functions (0.9.0)"
