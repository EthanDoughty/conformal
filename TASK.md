# Task: Path-Sensitive Control Flow Analysis for Shape Inference

## Goal
Improve precision of shape inference at if/else join points by tracking simple conditions that constrain dimensions, eliminating unnecessary losses to `unknown` when branch conditions provide shape information.

## Scope
- **Files to modify**:
  - `analysis/analysis_ir.py`: Extend If statement handling (lines 79-93)
  - `runtime/env.py`: Add conditional refinement capabilities
  - `runtime/shapes.py`: Add shape refinement predicates

- **Behaviors to add**:
  - Extract dimension-constraining conditions from If.cond expressions
  - Refine shapes in then-branch when condition implies shape properties
  - Refine shapes in else-branch when negated condition implies shape properties
  - Preserve refined shapes through join when both branches assign compatible shapes

- **Supported conditions** (pragmatic subset):
  - Scalar comparisons with dimension-related variables: `n > 0`, `m == k`
  - `isscalar(x)` and negation `~isscalar(x)` (if/when added to parser)
  - Direct equality checks: `x == y` where x or y are dimension variables

## Non-goals
- Full symbolic execution or constraint solving
- Tracking complex logical combinations (&&, ||, nested conditions)
- Inter-procedural analysis or function call refinement
- Refinement through while loops or for loops
- Path explosion (tracking multiple paths through program)
- Adding new MATLAB functions (isscalar, size, etc.) beyond what parser supports

## Invariants Impacted
- **IR analyzer authoritative**: Preserved (all changes in IR pipeline)
- **Best-effort analysis**: Preserved (refinement failures fall back to join)
- **Single-pass analysis**: Preserved (no fixed-point iteration)
- **Conservative soundness**: Tightened (refinement must not introduce false negatives)
- **Test expectations**: May change for tests that benefit from refinement (document which)

## Acceptance Criteria
- [ ] If condition `n > 0` allows preserving concrete dimension `n` in then-branch
- [ ] If condition `m == k` allows treating m and k as equivalent dimensions in then-branch
- [ ] Join of compatible refined shapes preserves precision (e.g., both branches assign `matrix[3 x 4]` → `matrix[3 x 4]`, not `matrix[3 x unknown]`)
- [ ] Unsupported conditions fall back to current join behavior (no regression)
- [ ] All 28 existing tests pass: `python3 mmshape.py --tests`
- [ ] New test demonstrates path-sensitive improvement over current behavior

## Commands to Run
```bash
# Full test suite
python3 mmshape.py --tests

# Run new path-sensitive test
python3 mmshape.py tests/test29.m

# Compare before/after on control flow tests
python3 mmshape.py tests/test10.m
python3 mmshape.py tests/test11.m

# Verify no regressions
make test
```

## Tests to Add/Change

**New test: tests/test29.m** — Path-sensitive if/else with concrete assignments
- Setup: No condition, just branches with different but compatible shapes
- Then branch: `A = zeros(3, 4)`
- Else branch: `A = zeros(3, 4)`
- Expected: `A = matrix[3 x 4]` (not `matrix[3 x unknown]`)
- Asserts: warnings=0, A=matrix[3 x 4]
- **Motivation**: Demonstrates join improvement when both branches produce identical shapes

**New test: tests/test30.m** — Path-sensitive with dimension variable condition
- Setup: `n = 4; k = 4;`
- Condition: `if n == k`
- Then branch: `A = zeros(n, k)` → should infer both are same symbolic value
- Else branch: `A = zeros(2, 2)`
- Expected: Join should preserve `n x n` or fall back gracefully
- Asserts: warnings=0, A=matrix (shape depends on join strategy)
- **Motivation**: Demonstrates dimension equivalence tracking

**Existing test changes**: None expected, but document if any test expectations improve due to refinement (e.g., if test10.m or test11.m benefit)

## Implementation Strategy

### Phase 1: Condition Analysis
1. Add `extract_condition_facts(cond: Expr) -> ConditionFacts` to collect:
   - Variable equality constraints: `n == k` → track n and k as equivalent
   - Positive comparisons: `n > 0`, `m >= 1` → track variable is positive/concrete
   - Scalar predicates: `isscalar(x)` → track x must be scalar shape

2. `ConditionFacts` dataclass contains:
   - `equalities: List[Tuple[str, str]]` — pairs of equivalent variables
   - `positive_vars: Set[str]` — variables known to be positive
   - `scalar_vars: Set[str]` — variables constrained to be scalar

### Phase 2: Environment Refinement
3. Add `Env.refine_with_facts(facts: ConditionFacts) -> Env` that:
   - Returns new environment with shapes refined by condition facts
   - For equalities, unify symbolic dimensions in existing shapes
   - For positive_vars, strengthen `None` dimensions to symbolic names
   - For scalar_vars, check consistency and narrow shapes

### Phase 3: Join Improvement
4. Modify `join_shape` or add `join_shape_refined` that:
   - Detects when both shapes are structurally identical (even if reached via different paths)
   - Preserves concrete information when safe
   - Falls back to current join_dim behavior when uncertain

### Phase 4: Integration
5. Modify `analyze_stmt_ir` If case (lines 79-93):
   - Extract condition facts from `stmt.cond`
   - Refine then_env with facts, else_env with negated facts (if safe)
   - Use refined join strategy
   - Keep fallback to current behavior if refinement fails

## Risks and Mitigations

**Risk**: Refinement introduces unsoundness (false negatives)
- **Mitigation**: Conservative extraction of facts; when uncertain, skip refinement

**Risk**: Performance degradation on large programs
- **Mitigation**: Refinement is shallow (no constraint solving); O(1) per condition

**Risk**: Breaking existing test expectations
- **Mitigation**: Refinement only improves precision; falling back to current join is always safe

**Risk**: Complexity creep toward full symbolic execution
- **Mitigation**: Strict scope limits; only handle enumerated condition patterns

**Risk**: Join logic becomes harder to understand
- **Mitigation**: Keep refinement separate from base join; clear documentation

## Open Questions (for mentor-reviewer)
1. Should we preserve concrete dimensions (`matrix[3 x 4]`) when both branches assign identical shapes, or always fall back to symbolic join?
2. How aggressively should we unify symbolic dimensions on equality conditions (e.g., `n == k`)?
3. Should negated conditions (`else` branch) receive refinement, or only positive conditions (`then` branch)?
4. What should happen when refinement facts conflict with existing environment shapes?
