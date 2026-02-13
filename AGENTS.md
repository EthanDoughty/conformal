# AGENTS.md — Orchestrator Reference

This file is for the **orchestrator (Coop)** and human integrator. Individual agents get project context from CLAUDE.md (auto-loaded) and role context from their own config in `.claude/agents/`.

## Agent Architecture

9 specialized agents with non-overlapping responsibilities, defined in `.claude/agents/`.

## Agent Roles and Responsibilities

### Planning & Review

#### spec-writer
- **Purpose**: Plans non-trivial changes before implementation
- **Responsibilities**:
  - Turns rough ideas into crisp TASK.md specifications
  - Defines Goal/Scope/Invariants and acceptance criteria
  - Identifies edge cases and architectural constraints
  - Plans test strategy
- **Triggers**: Before any non-trivial implementation
- **Output**: Complete TASK.md with clear specification
- **Does NOT**: Implement code, run tests

#### mentor-reviewer
- **Purpose**: Reviews designs, diffs, and patches with expert guidance
- **Responsibilities**:
  - Asks 3-7 substantive questions about intent, invariants, edge cases
  - Flags soundness issues, correctness concerns, invariant violations
  - Suggests targeted experiments to increase confidence
  - Provides mental model of what the change does
  - Separates REQUIRED (correctness) from OPTIONAL (quality) feedback
- **Triggers**: After spec-writer completes TASK.md, before implementation
- **Output**: Approved approach or questions needing answers
- **Does NOT**: Implement code (unless explicitly asked)

### Implementation & Fixing

#### implementer
- **Purpose**: Implements minimal, correct code changes for defined tasks
- **Responsibilities**:
  - Reads TASK.md (must have Goal/Scope/Invariants filled)
  - Produces minimal diffs (no unnecessary refactoring)
  - Runs tests after implementation
  - Reports results and status
- **Triggers**: After mentor-reviewer approves approach
- **Output**: Code changes + test results
- **Does NOT**: Plan features, change scope, refactor unnecessarily

#### test-fixer
- **Purpose**: Reactively fixes failing tests with minimal patches
- **Responsibilities**:
  - Analyzes test failures (diagnostics + root cause)
  - Produces minimal fix to restore tests to green
  - Re-runs tests to confirm fix
- **Triggers**: After test failures detected
- **Output**: Minimal patch + passing tests
- **Does NOT**: Add features, refactor, implement new functionality

### Quality & Validation

#### quality-assurance
- **Purpose**: Maintains code quality, consistency, and project hygiene
- **Responsibilities**:
  - Code style & PEP 8 compliance
  - Documentation completeness (docstrings, comments)
  - Code hygiene (dead code, unused imports, magic numbers)
  - Project consistency (naming conventions, patterns)
  - Best practices (DRY, function length, complexity)
  - Version control hygiene (no debug statements, credentials)
- **Triggers**: After implementer produces code, before merge, periodic maintenance
- **Output**: QUALITY ASSURANCE REPORT (quality score + issues)
- **Does NOT**: Check semantic correctness or test infrastructure

#### structural-ci-gatekeeper
- **Purpose**: Validates test infrastructure and CLI mechanics
- **Responsibilities**:
  - Test discovery (`glob("tests/**/*.m", recursive=True)` finds all tests)
  - CLI mechanics (`--tests`, `--strict`, `--compare` work correctly)
  - Exit codes (correct for success/failure)
  - Warning code stability (all use `W_*` prefix)
  - Expectation parsing (`% EXPECT:` format valid)
  - Determinism (same input → same output)
- **Triggers**: After code changes, before merge, version bump
- **Output**: STRUCTURAL CI REPORT (pass/fail for infrastructure)
- **Does NOT**: Validate semantic correctness of analysis

#### semantic-differential-auditor
- **Purpose**: Validates semantic correctness of shape inference
- **Responsibilities**:
  - Shape inference correctness (scalars, matrices, dimensions)
  - Symbolic dimension tracking (`n`, `m`, `k+m`, etc.)
  - Control flow analysis (if/else joins, while loops)
  - Operation semantics (matrix mult, concat, transpose)
  - Differential analysis (IR vs legacy, before vs after)
- **Triggers**: After code changes, before merge
- **Output**: SEMANTIC ANALYSIS REPORT (pass/fail for correctness)
- **Does NOT**: Check test infrastructure or CLI mechanics

### Operations

#### documentation-maintainer (proactive)
- **Purpose**: Keeps documentation synchronized with code after feature changes
- **Responsibilities**:
  - Update CHANGELOG.md, README.md, CLAUDE.md after code changes
  - Use change-type→document mapping to decide what needs updating
- **Triggers**: Proactively after implementer adds features, builtins, or shape rules
- **Output**: CHANGES MADE + VERIFICATION + SYNC STATUS
- **Does NOT**: Validate docstrings/comments (QA's job), change code behavior

#### release-coordinator (proactive)
- **Purpose**: Final pipeline step — organizes commits, handles versioning
- **Responsibilities**:
  - Organize changes into logical commits (feature, docs, infrastructure)
  - Propose version numbers (decimal: 0.8.5, not semver)
  - Draft CHANGELOG entries
  - Run pre-commit validation (`python3 mmshape.py --tests`)
  - **ALWAYS asks user approval** before committing
- **Triggers**: Proactively after QA and doc-maintainer complete — final step before user review
- **Output**: CHANGES SUMMARY + PROPOSED COMMITS + VERSION + CHANGELOG DRAFT + STATUS
- **Does NOT**: Implement features, modify code files, push without approval

### Project Governance

#### Integrator (Human)
- **Purpose**: Final decision making and integration
- **Responsibilities**:
  - Reviews all agent outputs
  - Approves or rejects plans
  - Approves release changes (CHANGELOG, version numbers)
  - Decides direction and priorities
  - Merges changes to repository
  - Creates git tags and pushes releases
- **Authority**: Final say on all decisions

#### Local backend: Ollama (tools/ai_local.py)
- Model: Qwen 2.5 Coder 14B (RTX 5070, 12K context, 85% GPU)
- Used for: file summaries, draft generation, single-file Q&A, boilerplate, docstrings
- Supports: `--compact`, `--summarize`, `--lite-context`, `--file path:N-M` line ranges
- Used by: spec-writer, mentor-reviewer, implementer (drafts only)
- Not used for: structural-ci-gatekeeper, semantic-differential-auditor

## Agent Invocation

All agents are invoked within Claude Code:

```
# Natural language
"Use the spec-writer agent to plan adding transpose support"
"Use the structural-ci-gatekeeper to validate the infrastructure"

# Via Task tool (in Claude Code internals)
Task(subagent_type="spec-writer", prompt="...", description="...")
```

## Agent Rules

### General Rules
- Agents get project context from CLAUDE.md (auto-loaded) — they do NOT need to read AGENTS.md
- Agents read TASK.md when implementing or reviewing changes
- Core invariant: **IR analyzer is authoritative**
- Minimal diffs; all tests must pass before claiming "done"

### Specific Rules
- **implementer**: Must not start work unless TASK.md has Goal/Scope/Invariants filled
- **mentor-reviewer**: Must ask 3-7 substantive questions before approval
- **quality-assurance**: Runs after code changes, focuses on quality not correctness
- **test validation**: Must use BOTH structural-ci-gatekeeper AND semantic-differential-auditor for comprehensive coverage
- **test-fixer**: Only produces minimal patches; does not add features or refactor
- **documentation-maintainer**: Proactively syncs docs after feature changes (CHANGELOG, README, CLAUDE.md)
- **release-coordinator**: Proactively organizes commits at end of pipeline; ALWAYS asks user approval before committing

## Pipeline Workflows

### Workflow 1: Complex Feature Development

```
1. Human: "Add element-wise multiplication (.* operator)"
   ↓
2. spec-writer: Creates TASK.md with Goal/Scope/Invariants
   ↓
3. mentor-reviewer: Reviews plan, asks 3-7 questions
   ↓
4. Human: Answers questions → mentor-reviewer approves
   ↓
5. implementer: Implements code, runs initial tests
   ↓
6. quality-assurance (proactive): Checks code quality and style
   ↓
7. documentation-maintainer (proactive): Syncs docs with changes
   ↓
8. release-coordinator (proactive): Organizes commits, proposes version
   ↓
9. Human Integrator: Approves commits → MERGE
```

### Workflow 2: Quick Bug Fix

```
1. Human: "Fix parser bug in line 42"
   ↓
2. implementer: Reads context, produces fix, runs tests
   ↓
3. quality-assurance (proactive): Quick quality check
   ↓
4. release-coordinator (proactive): Organizes commit
   ↓
5. Human Integrator: Approves → MERGE
```

### Workflow 3: Architecture Review Only

```
1. Human: "Should we refactor the shape domain?"
   ↓
2. mentor-reviewer: Asks questions, explores tradeoffs
   ↓
3. Human + mentor-reviewer: Iterate on design
   ↓
4. spec-writer: Documents final approach in TASK.md
   ↓
5. Human: Decides whether to proceed to implementation
```

### Workflow 4: Test Failure Recovery

```
1. Human: Runs tests → failures detected
   ↓
2. test-fixer: Analyzes failures, produces minimal patch
   ↓
3. quality-assurance: Checks patch quality (auto-triggered)
   ↓
4. structural-ci-gatekeeper + semantic-differential-auditor: Validate
   ↓
5. All green → Human: MERGE
```

### Workflow 5: Version Release (Standalone)

```
1. Human: "Prepare release for v0.8"
   ↓
2. release-coordinator: Reviews all changes since last version
   ↓
3. release-coordinator: Proposes commits, version, CHANGELOG
   ↓
4. Human: Approves → release-coordinator commits
   ↓
5. Human: Tags and pushes
```

For project architecture, commands, test format, shape system, and known behaviors, see **CLAUDE.md** (auto-loaded for all agents).
