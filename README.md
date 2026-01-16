# Mini-MATLAB Static Shape & Dimension Analysis
**Ethan Doughty**

This project implements a static shape and dimension analysis for a subset of the MATLAB programming language, referred to as Mini-MATLAB.

The goal of the analysis is to detect common matrix-related errors before runtime, using a custom parser and static analyzer designed specifically for MATLAB-style matrix semantics. The tool reasons about matrix shapes, symbolic dimensions, and control flow without relying on the MATLAB runtime.

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
- **Matrix–scalar comparisons** (`A == 0`)
- **Logical operators on non-scalars** (`&&`, `||`)
- **Incompatible variable reassignments**
- **MATLAB-aware fix suggestions**
  - e.g. suggesting `.*` instead of `*`

All warnings are reported with source line numbers, and the analysis continues in a "best-effort" manner even after detecting errors.

## Language Subset Design

The language subset and analysis design were chosen to isolate a subset of MATLAB that is dense enough to show interesting behaviors, but small enough to analyze with a custom static tool.

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

## Test Suite

The project includes a self-checking test suite consisting of 18 Mini-MATLAB programs.

Each test file:
- documents its intent using MATLAB comments
- declares expected warnings and final shapes using inline assertions:
  ```matlab
  % EXPECT: warnings = 1
  % EXPECT: A = matrix[n x (k+m)]

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
| 15    | Symbolic Concatenation         
| 16   | Row Slice Indexing
| 17   | Column and Full Slice Indexing
| 18   | Invalid indexing on scalars


For more information on the specifics of each of the test cases, see the tests/testN.m files. MATLAB comments are provided to describe intended behavior for each test case, and the reasoning for why the assertions pass.

## HOW TO RUN!

Run the full test suite: `python3 run_all_tests.py`

This script runs the analysis on `tests/test1.m` through `tests/test15.m`, prints any dimension warnings, shows the final environment, and checks all inline expectations automatically

To run a single test: `python3 runner.py tests/testN.m`

## Notes and Challenges

- The AST is represented using list-based nodes (e.g. ['assign', line, name, expr]), which makes it easy to implement analyses over the tree.

- Matrix literals are parsed with MATLAB-aware rules.

- The analyzer uses best-effort inference; Even when a definite mismatch is detected, it continues analysis to provide as much information as possible.

- The analyzer is strict on provable dimension errors. When an operation is definitely invalid (e.g., inner-dimension mismatch in A*B), it emits a warning and treats the expression result as unknown.

## Motivation and Future Directions

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.

Possible future extensions include:

- user-defined functions and interprocedural shape inference
- stricter invalidation semantics for definite errors
- richer symbolic constraint solving
- IDE or language-server integration
