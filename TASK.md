# TASK.md (ACTIVE)

## Branch
agent/<id>-<slug>

## Goal
<1–2 sentences describing what we’re changing and why.>

## Scope
- ✅ In scope:
  - <bullet>
  - <bullet>
- ❌ Out of scope:
  - <bullet>

## Invariants (must hold)
- IR analyzer remains authoritative; legacy is regression-only.
- Minimal diffs; no broad refactors unless explicitly requested.
- All tests pass: `python3 mmshape.py --tests`
- If parsing/recovery touched: add at least one targeted test that would have failed before.
- Never commit to main directly

## Acceptance Criteria (done when)
- [ ] Implementation matches goal + scope.
- [ ] Tests added/updated as needed.
- [ ] `python3 mmshape.py --tests` passes.
- [ ] Any new warning codes are `W_*` prefixed and stable.
- [ ] Reviewer questions answered (recorded below).

## Commands to run
- `python3 mmshape.py --tests`
- (optional) `python3 mmshape.py --compare tests/testN.m`
- (optional) `python3 mmshape.py <file>.m`

## Notes / Context
- Relevant modules: <e.g., frontend/matlab_parser.py, frontend/lower_ir.py, analysis/analysis_ir.py>
- Risks to watch: <lexer precedence, delimiter sync, join behavior, etc.>

## Reviewer Questions + Answers (fill during review)
1. Q:
   A:
2. Q:
   A: