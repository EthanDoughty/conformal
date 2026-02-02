# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project implements a static shape and dimension analysis for Mini-MATLAB, a subset of the MATLAB programming language. The analyzer detects matrix-related errors before runtime by reasoning about matrix shapes, symbolic dimensions, and control flow.

## Commands

### Running Tests
- **Full test suite**: `python3 run_all_tests.py` or `make test`
  - Runs all 21 test files (tests/test1.m through tests/test21.m)
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

### Other Commands
- **Clean build artifacts**: `make clean`
  - Removes `__pycache__` directories and `.pyc` files

## Architecture

### Three-Stage Pipeline

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

### Runtime Components (`runtime/`)

- `shapes.py`: Shape abstract domain
  - `Shape`: scalar | matrix[r x c] | unknown
  - `Dim`: int | str (symbolic name) | None (unknown)
  - Functions: `join_dim`, `dims_definitely_conflict`, `add_dim`

- `env.py`: Environment for variable bindings
  - `Env`: Maps variable names to shapes
  - `join_env`: Merges environments from control flow branches

### Legacy Code (`legacy/`)

Contains the original syntax-based analyzer. The IR-based analyzer (`analysis_ir.py`) is now the default and source of truth for test expectations.

## Shape System

The analyzer assigns each expression a shape from this abstract domain:

- **scalar**: Single values (e.g., `5`, `x`, scalar variables)
- **matrix[r x c]**: Matrices where `r` and `c` can be:
  - Concrete integers (e.g., `3`, `4`)
  - Symbolic names (e.g., `n`, `m`, `k`)
  - Unknown (`None`)
- **unknown**: When shape cannot be determined

### Key Analysis Features

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

The test runner parses these expectations and validates them against the analysis results. All test files are in `tests/` and numbered test1.m through test21.m.

## Important Implementation Details

### AST Representations

The project has two AST formats:

1. **Syntax AST** (list-based): `['assign', line, name, expr]`
   - Original parsed format from `matlab_parser.py`
   - Used by legacy analyzer

2. **IR AST** (dataclass-based): `Assign(line=line, name=name, expr=expr)`
   - Typed, structured representation from `ir/ir.py`
   - Used by current analyzer (`analysis_ir.py`)

The lowering pass (`frontend/lower_ir.py`) converts syntax AST → IR AST.

### Analysis Modes

- **Default**: IR-based analysis (`analyze_program_ir`)
- **Legacy**: Syntax-based analysis (`analyze_program_legacy`)
- **Compare mode**: Runs both and reports differences

The IR analyzer is the source of truth for test expectations.

### Best-Effort Analysis

The analyzer continues after detecting errors to provide maximum information. When a definite mismatch is detected (e.g., inner dimension mismatch in `A*B`), it emits a warning and treats the result as `unknown` to allow analysis to continue.

## Entry Points

- **CLI**: `mmshape.py` - Main command-line interface
- **Test Runner**: `run_all_tests.py` - Self-checking test suite
- **Analysis Functions**:
  - `analyze_program_ir(program: Program) -> (Env, List[str])` in `analysis/analysis_ir.py`
  - `analyze_program_legacy(ast) -> (Env, List[str])` in `legacy/analysis_legacy.py`
