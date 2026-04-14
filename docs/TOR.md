# Tool Operational Requirements

**Tool:** Conformal Static Shape Analyzer for MATLAB  
**Qualification level:** TQL-5 (DO-330 / DO-178C)  
**Document applies to:** v3.9.0 and later

---

## 1. Purpose and Context

Conformal is a static analysis tool that detects matrix dimension mismatches, type errors, and index-out-of-bounds conditions in MATLAB source code without executing the code and without requiring a MATLAB license. It is intended for use in verification workflows where manual review of dimension correctness is otherwise required.

Under DO-178C Table A-7, a tool qualifies at TQL-5 when its output is used to assist verification activities but does not replace any required verification step. Conformal meets this criterion: a clean run reduces the burden of manual dimension-correctness review but does not substitute for peer review, test execution, or structural coverage analysis.

---

## 2. Operational Requirements

The following requirements are numbered for traceability. Each is covered by at least one `% EXPECT_WARNING` or `% EXPECT_NO_WARNING` test annotation in the test suite.

**TOR-01.** The tool shall emit `W_INNER_DIM_MISMATCH` when the inner dimensions of a matrix multiplication `A * B` are provably unequal and both operands have known shapes.

**TOR-02.** The tool shall emit `W_ELEMENTWISE_MISMATCH` when the dimensions of an element-wise operation (`.*`, `./`, `.^`, `+`, `-`) are provably unequal and neither operand is scalar.

**TOR-03.** The tool shall emit `W_HORZCAT_ROW_MISMATCH` and `W_VERTCAT_COL_MISMATCH` when the conformability requirement for horizontal or vertical concatenation is provably violated.

**TOR-04.** The tool shall emit `W_RESHAPE_MISMATCH` when the element count of the input and output of a `reshape` call are provably unequal.

**TOR-05.** The tool shall emit `W_INDEX_OUT_OF_BOUNDS` when a subscript interval is provably outside `[1, dim]` for a matrix with a known concrete dimension.

**TOR-06.** The tool shall emit `W_FUNCTION_ARG_COUNT_MISMATCH` when a function is called with more arguments than its declaration accepts, excluding functions with `varargin` and classdef methods with implicit `self`.

**TOR-07.** The tool shall emit `W_UNKNOWN_FUNCTION` for any function call that is neither a recognized builtin nor resolved in the workspace scan.

**TOR-08.** The tool shall not suppress analysis after encountering an unresolvable expression; it shall fall back to `UnknownShape` and continue.

**TOR-09.** The tool shall produce identical output for the same source file, version, and flags across repeated invocations (determinism requirement).

**TOR-10.** The tool shall emit a `W_UNSUPPORTED_STMT` diagnostic for syntax it cannot analyze, rather than silently treating the statement as correct.

**TOR-11.** When invoked with `--format sarif`, the tool shall produce SARIF 2.1.0 JSON conforming to the OASIS schema, including per-artifact SHA-256 hashes of the analyzed source files.

**TOR-12.** The tool shall respect inline suppression directives (`% conformal:disable W_CODE`) and not suppress diagnostics for codes not named in the directive.

---

## 3. Analysis Scope and Exclusions

These exclusions define the boundary of the tool's guarantees. A clean run does not imply freedom from the error classes listed here.

**3.1 Unknown-shape propagation.** When a function call is unresolved, its return value receives `UnknownShape`. Downstream operations on that value will not produce dimension warnings. The **shape coverage metric** (visible in CLI output and in `run.properties` of SARIF output) quantifies this gap per file: `shapeCoverage.rate` is the fraction of final-environment variables with tracked shapes.

**3.2 N-dimensional arrays.** Conformal tracks matrices as 2-D (`rows x columns`). Operations on tensors with three or more dimensions are handled conservatively. Results are `UnknownShape` and no dimension checking is performed on the higher dimensions.

**3.3 Runtime-dependent shapes.** Shapes that depend on runtime values not statically determinable (file I/O results, user input, MEX function returns, `eval`, `evalin`) are unknown. `eval` and `evalin` content is never analyzed.

**3.4 Dynamic dispatch.** `feval` with a non-literal function name and function handles stored in data structures are not resolved. Operator overloading in complex inheritance hierarchies may not match MATLAB runtime behavior.

**3.5 Numeric values.** Conformal tracks dimensions and shapes, not numeric values or floating-point precision. It will not detect overflow, underflow, loss of precision, or ill-conditioned matrices.

**3.6 Code reachability.** Code after an unconditional early `return` or `error()` call is not analyzed. All other branches of `if`, `switch`, and `try` constructs are analyzed.

**3.7 Inter-procedural depth.** Analysis is per-function. Summary shapes are inferred from each function's body, but the tool does not perform whole-program pointer analysis or alias analysis across call chains that span more than the local workspace.

---

## 4. Configuration Data

The tool's behavior is determined by the following flags. All flags must be recorded in any verification evidence that references a Conformal run.

| Flag | Effect |
|------|--------|
| (none) | Default mode: 36 warning codes active |
| `--strict` | Adds 11 informational codes (`W_UNSUPPORTED_STMT`, comparisons, approximation warnings) |
| `--coder` | Adds 6 MATLAB Coder compatibility codes (`W_CODER_*`) |
| `--fixpoint` | Enables widening-based loop fixpoint iteration for more precise loop-body shapes |
| `--fail-on-warnings` | Exit code 1 if any warnings are produced; enables use as a CI gate |
| `--format sarif` | Emit SARIF 2.1.0 instead of plain text; includes SHA-256 artifact hashes |
| `--batch <dir>` | Analyze all `.m` files in the directory; combined with `--format sarif` for multi-file SARIF |

A `.conformal.json` project file in any ancestor directory can set `strict`, `coder`, and `fixpoint` persistently. Its presence and content should be recorded alongside any verification evidence.

---

## 5. Verification Evidence

**5.1 Integration tests.** The test suite contains 552 `.m` files annotated with `% EXPECT_WARNING: W_CODE` and `% EXPECT_NO_WARNING` directives. These annotations are machine-readable verification contracts: each asserts that a specific diagnostic is or is not produced at a specific source location. All 552 tests pass on each release. The test runner exits with code 1 if any assertion fails.

**5.2 Property-based tests.** 28 FsCheck property tests validate the shape lattice algebraic laws: join commutativity, associativity, monotonicity, and ordering consistency, across randomly generated shapes and intervals. These run as part of `--tests`.

**5.3 SARIF traceability.** In `--format sarif` mode, each analyzed file is represented as an artifact with a SHA-256 hash of the source text. This binds the diagnostic output to the exact source content, satisfying DO-178C section 11.14 (traceability of verification results to source artifacts).

**5.4 Corpus validation.** Conformal is validated against a corpus of 15,085 real-world MATLAB files across 34 open-source projects in 14 technical domains (aerospace, biomedical, optimization, numerical, CFD, and others). Corpus sweep results, including per-project clean rates and false-positive patterns, are recorded in the project's roundtable reports.

---

## 6. Anomaly Record

Tool anomalies affecting the scope of guarantees are recorded here with resolution status. An anomaly is any behavior that could cause the tool to produce a false negative (missed real error) or a materially incorrect shape for downstream analysis.

---

**ANO-001:** Colon operator precedence inversion  
*Versions affected:* all releases before v3.9.0  
*Status:* Fixed in v3.9.0  
*Description:* The parser precedence table assigned `:` precedence level 7, above `*` (level 6) and `+/-` (level 5). MATLAB specifies that `:` has lower precedence than all arithmetic operators. As a result, any expression of the form `a:b*c:d` or `a:b+c:d` was parsed as `(a:b) * (c:d)` or `(a:b) + (c:d)`, producing a matrix multiply or add of two range vectors rather than a stepped range. This fired a false-positive `W_INNER_DIM_MISMATCH` or `W_ELEMENTWISE_MISMATCH` on the expression itself, and caused the downstream variable to receive `UnknownShape`. Any dimension check involving a variable derived from a stepped range expression was silently suppressed.  
*False-negative risk:* High for files using stepped ranges with arithmetic step expressions (`dx = h*2`, `x = 0:dx:L`). The shape of `x` and all variables whose size derived from `length(x)` was unknown, suppressing checks on downstream array operations.  
*Detection method:* Real-world corpus sweep (CFD category, 42.9% clean rate).

---

**ANO-002:** Vector reduction returning vector shape instead of scalar  
*Versions affected:* all releases before v3.9.0  
*Status:* Fixed in v3.9.0  
*Description:* The `handleReduction` function, shared by `sum`, `prod`, `mean`, `min`, `max`, `std`, `var`, `median`, `norm`, `any`, `all`, and related builtins, used the pattern `MatrixCols c -> Matrix(Concrete 1, c)` for single-argument calls. This pattern matches any matrix including row vectors `matrix[1 x N]` and column vectors `matrix[M x 1]`. In MATLAB, applying a reduction to a vector returns a scalar; the pattern was correct only for general matrices with both dimensions greater than 1. As a result, `max(v)` where `v` was a row or column vector returned the vector's shape rather than `Scalar`.  
*False-negative risk:* Moderate. Any operation using `max(v)` or `sum(v)` as a scalar operand would carry the wrong shape. In code where the result was used in scalar arithmetic (`dt = 4/umax^2`), the wrong shape could trigger a false-positive or cause the downstream variable to receive `UnknownShape`, suppressing further checks.  
*Detection method:* Real-world corpus sweep (CFD category). Reproduced with a minimal test case.

---

## 7. Qualification Basis

Conformal is classified at TQL-5 under DO-330 section 4.2.

*Criterion:* The tool produces output used in a verification activity (detecting dimension errors), but the activity is not solely dependent on the tool. A human reviewer remains responsible for confirming coverage, reviewing shape metrics, and exercising judgment on `UnknownShape` results. A failing CI gate from `--fail-on-warnings` is advisory: it catches detectable errors but does not certify absence of all dimension errors.

*Justification:* The tool cannot be misled into certifying correct code as erroneous in a way that suppresses required test execution. It can miss errors (false negatives) in the classes documented in Section 3, but those classes are bounded and disclosed. The anomaly record in Section 6 documents the process by which the tool's false-negative surface is actively reduced.

*Tool version identification:* Run `conformal --version` to obtain the version string. For builds from source, the git commit hash is available via `git rev-parse HEAD` in the repository root. The combination of version string, commit hash, and enabled flags constitutes the tool configuration identifier for DO-330 tool qualification records.
