# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Overview

This project implements a static shape and dimension analysis for Mini-MATLAB, a subset of the MATLAB programming language. The analyzer detects matrix-related errors before runtime by reasoning about matrix shapes, symbolic dimensions, and control flow.

## Agent Architecture

This project uses **9 specialized Claude Code agents** with distinct, non-overlapping responsibilities. All agents are defined in `.claude/agents/` and invoked within Claude Code.

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
  - Test discovery (`glob("tests/test*.m")` finds all tests)
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

#### documentation-maintainer
- **Purpose**: Keeps all documentation synchronized with code
- **Responsibilities**:
  - Sync CLAUDE.md, AGENTS.md, README.md with codebase
  - Validate docstrings match function signatures
  - Maintain CHANGELOG.md entries
  - Check cross-references and examples
  - Ensure inline comments are accurate
- **Triggers**: After agent changes, API changes, before version bump, monthly review
- **Output**: DOCUMENTATION SYNC REPORT (drift score + fixes)
- **Does NOT**: Change code behavior, make architectural decisions

#### release-coordinator
- **Purpose**: Orchestrates version bumps and release process
- **Responsibilities**:
  - Run all validation agents before release
  - Propose version number and CHANGELOG entries
  - Verify release checklist complete
  - Coordinate release workflow
  - **ALWAYS asks user approval** before changing commit messages or README.md
- **Triggers**: Before version bump, monthly release prep, on demand
- **Output**: RELEASE READINESS REPORT (status + proposed changes)
- **Does NOT**: Auto-commit, auto-tag, or auto-push without approval

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

#### Local backend: LM Studio (tools/ai_local.py)
- Used for: spec-writer, mentor-reviewer, implementer (drafts only)
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
- All agents must read AGENTS.md + TASK.md (BACKLOG is informational only)
- Agents must preserve the core invariant: **IR analyzer is authoritative**
- Minimal diffs only; avoid broad refactors unless explicitly requested
- All tests must pass before claiming "done"

### Specific Rules
- **implementer**: Must not start work unless TASK.md has Goal/Scope/Invariants filled
- **mentor-reviewer**: Must ask 3-7 substantive questions before approval
- **quality-assurance**: Runs after code changes, focuses on quality not correctness
- **test validation**: Must use BOTH structural-ci-gatekeeper AND semantic-differential-auditor for comprehensive coverage
- **test-fixer**: Only produces minimal patches; does not add features or refactor
- **documentation-maintainer**: Syncs docs after agent/API changes, before releases
- **release-coordinator**: ALWAYS asks user approval before changing commit messages or README.md

## Pipeline Workflows

### Workflow 1: Complex Feature Development

```
1. Human: "Add element-wise multiplication (.* operator)"
   ↓
2. spec-writer: Creates TASK.md with Goal/Scope/Invariants
   ↓
3. mentor-reviewer: Reviews plan, asks 3-7 questions
   ↓
4. Human: Answers questions in TASK.md
   ↓
5. mentor-reviewer: Approves approach
   ↓
6. implementer: Implements code, runs initial tests
   ↓
7. quality-assurance: Checks code quality, style, documentation
   ↓
8. structural-ci-gatekeeper: Validates infrastructure
   ↓
9. semantic-differential-auditor: Validates correctness
   ↓
10. If failures:
    test-fixer: Produces minimal patch
    Go to step 7
    ↓
11. Human Integrator: Reviews all validation → MERGE
```

### Workflow 2: Quick Bug Fix

```
1. Human: "Fix parser bug in line 42"
   ↓
2. implementer: Reads context, produces fix, runs tests
   ↓
3. quality-assurance: Quick quality check (style, hygiene)
   ↓
4. structural-ci-gatekeeper: Validates infrastructure
   ↓
5. semantic-differential-auditor: Validates correctness
   ↓
6. Human Integrator: Reviews → MERGE
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

### Workflow 5: Version Release

```
1. Human: "Prepare release for v0.8"
   ↓
2. release-coordinator: Runs all validation agents
   • quality-assurance
   • structural-ci-gatekeeper
   • semantic-differential-auditor
   • documentation-maintainer
   ↓
3. release-coordinator: Reports validation status
   ↓
4. If issues found:
   - Use appropriate agents to fix
   - Return to step 2
   ↓
5. release-coordinator: Proposes CHANGELOG entry
   ↓
6. AskUserQuestion: "Approve CHANGELOG?"
   ↓
7. Human: Reviews and approves (or edits)
   ↓
8. release-coordinator: Proposes version number
   ↓
9. AskUserQuestion: "Approve version v0.8.0?"
   ↓
10. Human: Approves
    ↓
11. release-coordinator: Updates CHANGELOG.md
    ↓
12. release-coordinator: Reports READY
    ↓
13. Human: Reviews, commits, tags, pushes
```

### Workflow 4: Test Failure Recovery

```
1. Human: Runs tests → 2 failures detected
   ↓
2. test-fixer: Analyzes failures, produces minimal patch
   ↓
3. quality-assurance: Checks patch quality
   ↓
4. structural-ci-gatekeeper: Validates infrastructure
   ↓
5. semantic-differential-auditor: Validates correctness
   ↓
6. All green → Human: MERGE
```

## Entry Points

- CLI: `mmshape.py`
- Full test runner: `run_all_tests.py`
- Default analyzer: `analysis/analysis_ir.py` (`analyze_program_ir`)
- Legacy analyzer: `legacy/analysis_legacy.py` (`analyze_program_legacy`)

## Commands

## Running Tests

- **Full test suite**: `python3 run_all_tests.py` or `make test`
  - Runs all test files (tests/test1.m through tests/test*.m)
  - Automatically validates inline expectations (comments starting with `% EXPECT:`)
  - Reports pass/fail status for each test

- **Single test**: `python3 mmshape.py tests/testN.m` or `make run FILE=tests/testN.m`
  - Runs analysis on a single test file
  - Displays warnings and final environment

- **Compare mode**: `python3 mmshape.py --compare tests/testN.m` or `make compare FILE=tests/testN.m`
  - Compares legacy syntax analyzer vs IR analyzer
  - Useful for debugging differences between the two pipelines

- **CLI test runner**: `python3 mmshape.py --tests`
  - Runs the full test suite via the CLI tool

## Other Commands
- **Clean build artifacts**: `make clean`
  - Removes `__pycache__` directories and `.pyc` files

## Architecture

## Three-Stage Pipeline

The analyzer uses a three-stage pipeline:

1. **Frontend** (`frontend/`): Parses Mini-MATLAB source → Syntax AST
   - `matlab_parser.py`: Lexer and recursive-descent parser
   - `lower_ir.py`: Lowers syntax AST to IR AST
   - `pipeline.py`: Convenience functions for the pipeline

2. **IR** (`ir/`): Intermediate representation with typed nodes
   - `ir.py`: Dataclass-based IR AST (Expr, Stmt, Program)
   - Clean, typed representation (vs. list-based syntax AST)

3. **Analysis** (`analysis/`): Static shape inference over IR
   - `analysis_ir.py`: Main IR-based analyzer (current default)
   - `analysis_legacy.py`: Legacy syntax-based analyzer (for comparison)
   - `analysis_core.py`: Shared logic for shape compatibility checks
   - `matrix_literals.py`: Matrix literal shape inference
   - `diagnostics.py`: Warning message generation

## Runtime Components (`runtime/`)

- `shapes.py`: Shape abstract domain
  - `Shape`: scalar | matrix[r x c] | unknown
  - `Dim`: int | str (symbolic name) | None (unknown)
  - Functions: `join_dim`, `dims_definitely_conflict`, `add_dim`

- `env.py`: Environment for variable bindings
  - `Env`: Maps variable names to shapes
  - `join_env`: Merges environments from control flow branches

## Legacy Code (`legacy/`)

Contains the original syntax-based analyzer. The IR-based analyzer (`analysis_ir.py`) is now the default and source of truth for test expectations.

## Shape System

The analyzer assigns each expression a shape from this abstract domain:

- **scalar**: Single values (e.g., `5`, `x`, scalar variables)
- **matrix[r x c]**: Matrices where `r` and `c` can be:
  - Concrete integers (e.g., `3`, `4`)
  - Symbolic names (e.g., `n`, `m`, `k`)
  - Unknown (`None`)
- **unknown**: When shape cannot be determined

## Key Analysis Features

- **Symbolic dimension tracking**: Variables like `n` and `m` represent dimensions
- **Symbolic arithmetic**: Concatenation produces dimensions like `n x (k+m)`
- **Control flow joins**: Merges shapes from `if`/`else` branches conservatively
- **Single-pass loop analysis**: Loops analyzed once (no fixed-point iteration)

## Test File Format

Test files use inline assertions in MATLAB comments:

```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
```

The test runner parses these expectations and validates them against the analysis results. All test files are in `tests/` and numbered test1.m through test27.m.

## Important Implementation Details

## AST Representations

The project has two AST formats:

1. **Syntax AST** (list-based): `['assign', line, name, expr]`
   - Original parsed format from `matlab_parser.py`
   - Used by legacy analyzer

2. **IR AST** (dataclass-based): `Assign(line=line, name=name, expr=expr)`
   - Typed, structured representation from `ir/ir.py`
   - Used by current analyzer (`analysis_ir.py`)

The lowering pass (`frontend/lower_ir.py`) converts syntax AST → IR AST.

## Analysis Modes

- **Default**: IR-based analysis (`analyze_program_ir`)
- **Legacy**: Syntax-based analysis (`analyze_program_legacy`)
- **Compare mode**: Runs both and reports differences

The IR analyzer is the source of truth for test expectations.

## Best-Effort Analysis

The analyzer continues after detecting errors to provide maximum information. When a definite mismatch is detected (e.g., inner dimension mismatch in `A*B`), it emits a warning and treats the result as `unknown` to allow analysis to continue.

## Known Behaviors
- Test discovery is dynamic in `run_all_tests.py` (`glob("tests/test*.m")`), so treat files in `tests/` as source of truth if docs disagree.
- `run_all_tests.py --compare` currently does not execute compare-mode output due to a wiring bug (the parsed flag is not used at runtime).
- `python3 mmshape.py --tests --compare` currently behaves like `--tests` only; compare mode is only reliable for single-file runs (`mmshape.py --compare tests/testN.m`).
- `--strict` fails if any `W_UNSUPPORTED_*` warning is emitted. This is expected for unsupported-construct recovery tests: `tests/test22.m`, `tests/test23.m`, `tests/test24.m`, `tests/test25.m`, and `tests/test27.m`.
- IR analysis is the default for CLI analysis (`mmshape.py`) and for test expectations, while legacy analysis remains available for per-file comparison.
