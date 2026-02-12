# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project implements a static shape and dimension analysis for Mini-MATLAB, a subset of the MATLAB programming language. The analyzer detects matrix-related errors (dimension mismatches, incompatible operations) before runtime by reasoning about matrix shapes, symbolic dimensions, and control flow without executing code.

**Key principle**: The IR-based analyzer (`analysis/analysis_ir.py`) is the authoritative implementation. The legacy syntax-based analyzer exists only for regression comparison.

## Essential Commands

**Run all tests**:
```bash
make test                          # or: python3 run_all_tests.py
python3 mmshape.py --tests         # via CLI
```

**Analyze a single file**:
```bash
make run FILE=tests/test4.m        # or: python3 mmshape.py tests/test4.m
```

**Compare legacy vs IR analyzer** (single file only):
```bash
make compare FILE=tests/test4.m    # or: python3 mmshape.py --compare tests/test4.m
```

**Strict mode** (fail on unsupported constructs):
```bash
python3 mmshape.py --strict tests/test22.m
```

**Clean build artifacts**:
```bash
make clean
```

## Local LLM Delegation (Token Conservation)

A local Qwen 2.5 Coder 14B model (Ollama, RTX 5070) is available as a **token conservation tool**. Claude should delegate to it aggressively for any task that involves a single file and does not require multi-file reasoning or tool use. Every token the local LLM handles is a token Claude doesn't spend.

**Delegation is the default for these tasks**:
- Summarizing a file before deciding whether to read it fully
- Drafting docstrings, comments, or boilerplate code
- Answering questions about a single file's contents
- Validating assumptions about a module's API or behavior
- Generating test case skeletons or code templates
- Quick "what does this function do?" queries

**Claude delegates via Bash**:
```bash
# Summarize a file (cached — instant on repeat calls)
python3 tools/ai_local.py --summarize --file runtime/shapes.py

# Ask about specific lines (saves tokens on large files)
python3 tools/ai_local.py --file frontend/matlab_parser.py:100-150 "what does this parsing rule handle?"

# Draft boilerplate with compact file attachment (strips comments/blanks)
python3 tools/ai_local.py --compact --file analysis/diagnostics.py "add a new warning function for transpose mismatch"

# Quick question with minimal context overhead
python3 tools/ai_local.py --lite-context --file ir/ir.py "list all Expr subclasses"

# General knowledge (no project context needed)
python3 tools/ai_local.py --no-context "explain lattice widening in abstract interpretation"
```

**Flags for token efficiency**:
| Flag | Effect |
|------|--------|
| `--file path:N-M` | Attach only lines N through M of a file |
| `--compact` | Strip comment lines and blank lines from attachments |
| `--lite-context` | Use ~400-token project summary instead of full docs (~7K tokens) |
| `--summarize` | Generate and cache file summaries (instant on repeat) |

**When NOT to delegate**: multi-file reasoning, correctness-critical analysis (soundness of joins, symbolic arithmetic edge cases), anything requiring codebase search or tool use, tasks needing >12K tokens of context.

## Architecture

### Three-Stage Pipeline

1. **Frontend** (`frontend/`) — Parsing and IR lowering
   - `matlab_parser.py`: Lexer and recursive-descent parser → list-based syntax AST
   - `lower_ir.py`: Syntax AST → typed IR AST (dataclass-based)
   - `pipeline.py`: Convenience wrappers

2. **IR** (`ir/`) — Typed intermediate representation
   - `ir.py`: Dataclass definitions (Expr, Stmt, Program)
   - Clean, typed nodes vs. legacy `['assign', line, name, expr]` format

3. **Analysis** (`analysis/`) — Static shape inference
   - `analysis_ir.py`: **Main IR-based analyzer** (default, authoritative)
   - `analysis_legacy.py`: Legacy syntax-based analyzer (for comparison only)
   - `analysis_core.py`: Shared compatibility checks
   - `matrix_literals.py`: Matrix literal shape inference
   - `diagnostics.py`: Warning message generation

### Runtime Components (`runtime/`)

- `shapes.py`: Abstract shape domain
  - `Shape`: `scalar | matrix[r x c] | unknown`
  - `Dim`: `int | str (symbolic) | None (unknown)`
  - Key functions: `join_dim`, `dims_definitely_conflict`, `add_dim`

- `env.py`: Variable environment
  - `Env`: Maps variable names → shapes
  - `join_env`: Merges branches in control flow

### Shape System

Each expression gets a shape from this abstract domain:
- **scalar**: Single values (e.g., `5`, scalar variables)
- **matrix[r x c]**: Where `r` and `c` can be:
  - Concrete integers (e.g., `3`, `4`)
  - Symbolic names (e.g., `n`, `m`, `k`)
  - Unknown (`None`)
- **unknown**: When shape cannot be determined

**Key features**:
- Symbolic dimension tracking (e.g., `n`, `m` represent dimensions)
- Symbolic arithmetic for concatenation (e.g., `n x (k+m)`)
- Control flow joins (merges `if`/`else` branches conservatively)
- Single-pass loop analysis (no fixed-point iteration)

## Test File Format

Tests use inline assertions in MATLAB comments:
```matlab
% EXPECT: warnings = 1
% EXPECT: A = matrix[n x (k+m)]
```

The test runner (`run_all_tests.py`) validates these expectations against analysis results. All test files are in `tests/` (test1.m through test27.m, discovered via `glob("tests/test*.m")`).

## Critical Implementation Details

### Two AST Formats

1. **Syntax AST** (list-based): `['assign', line, name, expr]`
   - Original parsed format from `matlab_parser.py`
   - Used by legacy analyzer

2. **IR AST** (dataclass-based): `Assign(line=line, name=name, expr=expr)`
   - Typed representation from `ir/ir.py`
   - Used by current analyzer (authoritative)
   - Lowering: `frontend/lower_ir.py` converts syntax → IR

### Analysis Modes

- **Default**: IR-based (`analyze_program_ir`) — this is the source of truth
- **Legacy**: Syntax-based (`analyze_program_legacy`) — for regression only
- **Compare**: Runs both and reports differences (single-file only)

### Best-Effort Analysis

When a definite mismatch is detected (e.g., inner dimension mismatch in `A*B`), the analyzer:
1. Emits a warning
2. Treats the result as `unknown`
3. Continues analysis to provide maximum information

## Known Behaviors and Gotchas

- Test discovery is dynamic via `glob("tests/test*.m")` in `run_all_tests.py`
- `--compare` mode only works reliably for single-file runs; `--tests --compare` ignores the compare flag
- `--strict` mode fails if any `W_UNSUPPORTED_*` warning is emitted (expected for test22.m, test23.m, test24.m, test25.m, test27.m)
- When editing parser/lowering, check delimiter syncing and token precedence carefully

## Agent Workflow

**Read AGENTS.md for complete agent roles, responsibilities, and workflow pipelines.**

This project uses nine specialized Claude Code agents with non-overlapping responsibilities:

**Planning & Review**:
- spec-writer: Plans features before implementation
- mentor-reviewer: Reviews designs with expert guidance

**Implementation & Fixing**:
- implementer: Implements code changes
- test-fixer: Fixes test failures with minimal patches

**Quality & Validation**:
- quality-assurance: Maintains code quality, style, and project hygiene
- structural-ci-gatekeeper: Validates test infrastructure and CLI mechanics
- semantic-differential-auditor: Validates semantic correctness of analysis

**Operations**:
- documentation-maintainer: Keeps docs synchronized with code
- release-coordinator: Orchestrates version bumps and releases

All agents must preserve the core invariant: IR analyzer is authoritative.

## Project Invariants

- IR analyzer (`analysis_ir.py`) is the source of truth for test expectations
- Legacy analyzer exists only for regression comparison
- All tests must pass: `python3 mmshape.py --tests`
- New warning codes must use `W_*` prefix and be stable
- Minimal diffs; avoid broad refactors unless explicitly requested
- If touching parser/recovery, add at least one test that would have failed before
