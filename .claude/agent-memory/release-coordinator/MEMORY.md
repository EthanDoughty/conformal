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
- 0.8.5: reshape/repmat shape rules
- 0.8.4: Rich builtin shape rules (12 functions)
- 0.8.3: Extracted indexing helper
- 0.8.2: Unified Apply IR node
- 0.8.1: Expanded builtins to 19 functions

## Release Decision Criteria
- All tests pass (42/42 in both modes)
- Quality score not tracked (no QA agent ran)
- Documentation synchronized (user confirmed)
- Breaking changes: Requires major bump
- New feature + bug fix: Minor bump (precedent: 0.8.4 added features + fixed non-determinism)
