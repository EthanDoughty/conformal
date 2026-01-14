# Mini-MATLAB Static Shape & Dimension Analysis
**Ethan Doughty**

This project implements a **static shape and dimension analysis** for a subset of the MATLAB programming language (called **Mini-MATLAB**)
The goal of the analysis is to detect common matrix-related errors before runtime, using a custom static analyzer built specifically for a small but expressive MATLAB subset.

## What the Analysis Detects

The analyzer detects dimension mismatches in the following matrix operations:

- **matrix addition** (`+`, `-`)
- **matrix multiplication** (`*`)
- **elementwise operations** (`.*`, `./`, `/`)
- **vector transpose** (`v'`)
- **colon-generated vectors** (`1:n`)
- **scalar–matrix operations** (`s*A`, `s+A`)
- **indexing behavior** (`A(i,j)`)
- **matrix–scalar comparisons** (`A == 0`)
- **logical operators on non-scalars** (`&&, ||`)
- **incompatible variable reassignments**
- **fix suggestions** (e.g., suggesting .* instead of *)

All warnings are reported with source line numbers

This property is important because many MATLAB runtime errors come from incompatible dimensions. This analysis attempts to detect these errors statically, without needing to use the MATLAB runtime environment.

## Language Subset Design

The language subset and analysis design were chosen to isolate a subset of MATLAB that is dense enough to show interesting behaviors, but small enough to analyze with a custom static tool.

The subset includes:

- assignments
- expressions
- function calls (zeros, ones)
- control flow (if, for, while)
- symbolic dimensions
- indexing and transpose

Loops are analyzed conservatively using a single pass, which is sufficient for the test cases and keeps the analysis focused on shape reasoning rather than loop invariants.

## Test Suite

The tool was evaluated on **ten Mini-MATLAB programs**, each designed to test a specific aspect of the analysis:

| Test | Description                         | Expected Result |
|------|-------------------------------------|-----------------|
| 1    | Basic addition                      | OK              |
| 2    | Addition mismatch                   | Warning         |
| 3    | Valid matrix multiply               | OK              |
| 4    | Multiply mismatch                   | Warning         |
| 5    | Colon vector + transpose            | OK              |
| 6    | Symbolic dimension multiplication  | OK              |
| 7    | Scalar expansion                    | OK              |
| 8    | Elementwise mismatch                | Warning         |
| 9    | Indexing behavior                   | OK              |
| 10   | Symbolic mismatch in branch         | Warning         |


For more information on the specifics of each of the 10 test cases, see the testN.m files. MATLAB comments are provided to describe intended behavior for each test case, and the reasoning for why they pass or fail. 

## HOW TO RUN!
In the command line: `python3 run_all_tests.py`

This script runs the analysis on `tests/test1.m` through `tests/test10.m`, prints any dimension warnings, and then shows the final environment for each test

To run each test individually: `python3 runner.py tests/testN.m`

## Notes and Challenges:

The AST uses list structures (['assign', 'A', expr]), which made it easy to write an analysis over it

To keep the analysis understandable, loops are handled in a single pass. This is enough for the test cases, which focus on shape reasoning over complex loop invariants, but the decision was made to omitt this feature to keep the scope of the project less complex than it needed to be, and to not stray away from the original goal of the analysis.

Possible additions that could be included in the future (These features are pretty important in Matlab):
-defined matrix literals ([1 2; 3 4])
-more reasoning about loop iterations
-attaching line numbers to warnings

I felt that it was very rewarding to use MATLAB as the source language for a static analysis tool. I wanted to choose something that was niche enough to be interesting and unique, while also being relevant to modern topics. MATLAB was perfect for this, as it is a language used widely for its scientific computing ability compared to other languages.


