# Mini-MATLAB Static Shape & Dimension Analysis
**Ethan Doughty**

This project implements a static shape and dimension analysis for a carefully chosen subset of the MATLAB programming language, referred to as Mini-MATLAB.

The goal of the analysis is to detect common matrix-related errors before runtime, using a custom parser and static analyzer designed specifically for MATLAB-style matrix semantics. The tool reasons about matrix shapes, symbolic dimensions, and control flow without relying on the MATLAB runtime.

## Requirements

- **Python 3.10+**
- No third-party dependencies
- Tested on Linux
- No MATLAB installation required

## What the Analysis Detects

The analyzer statically detects dimension and shape issues in the following constructs:

- **Matrix addition and subtraction** (`+`, `-`)
- **Matrix multiplication** (`*`)
- **Elementwise operations** (`.*`, `./`, `/`)
- **Scalar–matrix operations** (`s*A`, `s + A`)
- **Matrix literals** (`[1 2; 3 4]`, `[A B]`, `[A; B]`)
- **Horizontal and vertical concatenation constraints**
- **Vector transpose** (`v'`)
- **Colon-generated vectors** (`1:n`)
- **MATLAB-style indexing and slices** (`A(i,j)`, `A(i,:)`, `A(:,j)`, `A(:,:)`)
- **Range indexing** (`A(2:5,:)`, `A(:,2:5)`)
- **Matrix–scalar comparisons** (`A == 0`)
- **Logical operators on non-scalars** (`&&`, `||`)
- **Incompatible variable reassignments**
- **MATLAB-aware fix suggestions**
  - e.g. suggesting `.*` instead of `*`

All warnings are reported with source line numbers, and the analysis continues in a "best-effort" manner even after detecting errors.

## Language Subset Design

The language subset and analysis design were chosen to isolate a fragment of MATLAB that is dense enough to show interesting behaviors, but small enough to analyze with a custom static tool.

The subset includes:

- assignments and expressions
- function calls (zeros, ones)
- control flow (if, for, while)
- symbolic dimensions
- indexing and transpose

Loops are analyzed conservatively using a single pass, which keeps the analysis focused on shape reasoning rather than loop invariants. This design choice is fine for the intended test cases and avoids unnecessary complexity.

## Shape System

Each expression is assigned a shape from a small abstract domain:

- `scalar`
- `matrix[r x c]` where `r` and `c` may be:
  - concrete integers
  - symbolic names (`n`, `m`, `k`)
  - unknown (`None`)
- `unknown`

The analysis supports:
- symbolic dimension equality
- symbolic dimension joins across control flow
- symbolic dimension addition for matrix concatenation (e.g. `n x (k+m)`)

## Project Structure

frontend/    Parsing and IR lowering
ir/          Typed IR dataclass definitions
analysis/    Shape analysis, diagnostics, and core semantics
legacy/      Original syntax-based analyzer (for comparison)
runtime/     Shape domain and environments
tests/       Self-checking Mini-MATLAB programs
tools/       Debugging utilities (AST printer)

## Test Suite

The project includes a self-checking test suite of Mini-MATLAB programs.

Each test file:
- documents its intent using MATLAB comments
- declares expected warnings and final shapes using inline assertions:
  ```matlab
  % EXPECT: warnings = 1
  % EXPECT: A = matrix[n x (k+m)]
  ```

| Tests | Category
|------|------------
| 1-4  | Basic arithmetic
| 5    | Colon vectors and transpose
| 6    | Symbolic dimensions
| 7    | Scalar Expansion
| 8    | Elementwise mismatches
| 9    | Indexing semantics
| 10   | Control-flow joins
| 11-14| Matrix literals
| 15   | Symbolic Concatenation
| 16-18| Indexing Slices and Invalid Scalar Indexing
| 19-20| Range Indexing Slices
| 21   | Invalid non-scalar index argument
| 22-25| Unsupported construct recovery (parse recovery tests)
| 26   | Extended control flow patterns
| 27   | Advanced symbolic dimension tracking
| 28   | Expanded builtins and unknown function warning
| 29   | Multiline matrix literals with newline separators
| 30   | Apply node disambiguation (colon/builtin/unbound/indexing)
| 31   | Rich builtin shape rules (eye/rand/randn/abs/sqrt/transpose/length/numel)

## Getting Started

Clone and Verify
`git clone <repo-url>`
`cd matlab-static-dimension-analysis`
`make test`

Analyze a File
`make run FILE=tests/test4.m`

Compare IR vs Legacy Analyzer
`make compare FILE=tests/test4.m`

Example Output
`Warnings:`
  `- Line 11: Dimension mismatch in expression (A * x):`
    `inner dims 4 vs 5 (shapes matrix[3 x 4] and matrix[5 x 1])`

`Final environment:`
`Env{A: matrix[3 x 4], x: matrix[5 x 1], y: unknown}`

## CLI Options

mmshape.py file.m – analyze a file (IR-based)

`--compare` – compare legacy vs IR analyzer

`--tests` – run full test suite

Exit codes:

0 – success

1 – parse error, analyzer mismatch, or test failure

## Notes and Challenges

- The IR enforces structural invariants absent in the raw syntax AST

- Matrix literals are parsed with MATLAB-aware rules.

- The analyzer uses best-effort inference; Even when a definite mismatch is detected, it continues analysis to provide as much information as possible.

- The analyzer is strict on provable dimension errors. When an operation is definitely invalid (e.g., inner-dimension mismatch in A*B), it emits a warning and treats the expression result as unknown.

## Limitations
This tool does not support: 
- user-defined functions
- cell arrays or structs
- strings
- file I/O
- plotting or graphics
- precise loop invariants

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

Possible future extensions include:

- user-defined functions and interprocedural shape inference
- stricter invalidation semantics for definite errors
- richer symbolic constraint solving
- IDE or language-server integration
