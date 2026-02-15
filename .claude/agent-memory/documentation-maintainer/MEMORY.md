# Documentation Maintainer Memory

## Common Drift Patterns

### Test Path References
- **Test organization**: Categorized subdirectories (basics/, symbolic/, indexing/, control_flow/, literals/, builtins/, recovery/)
- **Glob pattern**: tests/**/*.m (recursive) not tests/test*.m
- **Pattern**: When referencing tests, use descriptive category paths not numbered testN.m
- **Example paths**: tests/basics/valid_add.m, tests/recovery/struct_field.m, tests/builtins/constructors.m

### Version Documentation Flow
- **CHANGELOG.md**: Add new version entry with date
- **Format**: Keep a Changelog - sections: Added, Changed, Fixed, Removed
- **Detail Level**: Concise but specific (file names, warning codes, test numbers)
- **User Preference**: No Co-Authored-By lines in commits

## Project-Specific Conventions

### Warning Codes
- Prefix: `W_*` (e.g., W_UNKNOWN_FUNCTION, W_UNSUPPORTED_*)
- Document in: analysis/diagnostics.py (docstrings), CHANGELOG.md (when added)
- CLAUDE.md mentions warning code stability but does not list all codes

### Builtin Functions
- Source of truth: `KNOWN_BUILTINS` constant in analysis/builtins.py (centralized as of v0.8.6)
- Shape rules: `BUILTINS_WITH_SHAPE_RULES` in analysis/builtins.py
- **v0.8.7 milestone**: Full coverage (19/19 — KNOWN_BUILTINS == BUILTINS_WITH_SHAPE_RULES)
- CLAUDE.md and README.md do NOT list individual builtins (too volatile)
- Document new shape rules in CHANGELOG.md with function signatures
- New builtins added in Phase 3: det, diag, inv, linspace, norm

### Test Organization (Updated v0.8.5+)
- Discovery: Dynamic via `glob("tests/**/*.m", recursive=True)`
- Structure: Categorized subdirectories with descriptive names
- No more numbered test files - use category/descriptive_name.m format
- Example: tests/basics/valid_add.m not tests/test1.m

## Files to Always Check on Changes

### Parser/Frontend Changes
- CLAUDE.md: Architecture section (lines ~80-98)
- README.md: "What the Analysis Detects" (lines ~17-34)

### Test Suite Changes
- CLAUDE.md: Test file range (line ~134)
- README.md: Test count (line ~79) and test table (lines ~89-107)

### Warning/Diagnostic Changes
- CHANGELOG.md: Add entry for new warning codes
- analysis/diagnostics.py: Ensure docstrings mention warning codes

### Version Bumps
- CHANGELOG.md: Move items from Unreleased to new version section
- Use YYYY-MM-DD date format
- Keep Unreleased section for future changes
- Version format: `[0.8.2]` (no 'v' prefix in section headers)

### Symbolic Arithmetic Synchronization
- **CLAUDE.md line 122**: Lists symbolic arithmetic capabilities
- **README.md line 65**: Bulleted list of symbolic arithmetic features
- **Pattern**: When adding new symbolic operators (add_dim, mul_dim), update BOTH locations
- **v0.8.5 update**: Added multiplication support for repmat (mul_dim function)

## Recent Documentation Updates (v0.8.2-v0.8.5)

### v0.8.2 Architectural Change
- Unified Apply IR node for call-vs-index disambiguation
- Parser no longer distinguishes Call vs Index at parse time
- Analyzer disambiguates using colon heuristic and shape information
- Key insight: This is a major architectural shift (parse-time → analysis-time decision)
- Affects: README.md architecture mentions, test count (29→30)

### v0.8.3 Refactoring
- Extracted `_eval_indexing()` helper in analysis_ir.py
- Consolidated three copies of indexing logic
- Typed `Apply.args` properly as `List[IndexArg]`
- Pure refactor: no test expectation changes, no user-facing changes

### v0.8.4 Rich Builtin Shape Rules
- Added shape rules for 12 functions (up from ~4)
- Introduced `_eval_index_arg_to_shape()` helper
- Test31.m (now tests/builtins/constructors.m) with 38 assertions
- Fixed non-deterministic output in join_env() (sorted iteration)

### v0.8.5 Symbolic Multiplication
- Added `mul_dim()` function in runtime/shapes.py
- Shape rules for reshape and repmat
- Required updating symbolic arithmetic documentation in TWO places:
  - CLAUDE.md line 122 (Key features list)
  - README.md line 65 (analysis supports bullet list)
- New test: tests/builtins/reshape_repmat.m (7 assertions)
- Test count: 32 total

### v0.8.7 Complete Builtin Coverage (Phase 3)
- Shape rules for 5 new builtins: det, diag, inv, linspace, norm
- Fixed zeros(n)/ones(n) 1-arg gap (now returns n×n, was unknown)
- Achieved full coverage: BUILTINS_WITH_SHAPE_RULES == KNOWN_BUILTINS (19/19)
- New test file: tests/builtins/remaining_builtins.m (19 expectations: 1 warnings + 18 shapes)
- Test count: 44 total
- **Documentation impact**: CHANGELOG.md ✓, README.md ✓ (test table updated)

### v0.9.1 Dimension Arithmetic in Builtin Arguments
- Enhanced `expr_to_dim_ir` to handle BinOp (+, -, *) in dimension arguments
- Symbolic arithmetic now works in builtin calls: zeros(n+1, m), reshape(A, 2*n, m+1)
- Concrete arithmetic folding: zeros(2+3, 4*2) → matrix[5 x 8]
- New test file: tests/builtins/dim_arithmetic.m (9 cases, 0 warnings)
- Test count: 45 total
- **Documentation pattern**: Dimension capabilities appear in THREE places:
  1. CHANGELOG.md [Unreleased] — Feature description + test file
  2. CLAUDE.md line 122 — "Key features" symbolic arithmetic list
  3. README.md line 65 — "The analysis supports" bullet list
- **Key insight**: When symbolic dimension capabilities expand, check all three locations

### v0.9.2 Principled Loop Widening
- Replaced iteration-limit approach (MAX_LOOP_ITERATIONS=3) with widening-based fixpoint
- Added widening operators: `widen_dim()`, `widen_shape()`, `widen_env()` in runtime/
- 3-phase algorithm in `_analyze_loop_body()`: discover, stabilize, post-loop join
- Guarantees convergence in ≤2 iterations (vs unpredictable with iteration limit)
- Updated 3 existing tests (EXPECT_FIXPOINT warnings: 3→1 for growth patterns)
- Added 4 new loop tests: widen_col_grows, widen_multiple_vars, widen_self_reference, widen_while_growth
- Test count: 49 total
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.9.2] section with Added/Changed
  2. CLAUDE.md line 105 — Added widen_dim/widen_shape to key functions
  3. CLAUDE.md line 109 — Added widen_env description
  4. CLAUDE.md line 124-125 — Rewrote loop analysis description with widening details
  5. README.md line 49 — Rewrote loop analysis philosophy
  6. README.md line 68 — Added widening to symbolic capabilities list
- **Pattern**: Loop analysis changes affect README philosophy section + CLAUDE.md technical details

### v0.9.3 Shape.bottom() Lattice Fix
- Added `Shape.bottom()` as distinct lattice element (bottom = unbound/no-info, unknown = error/top)
- Fixed soundness bug: `widen_shape(matrix, unknown_error)` now correctly returns unknown instead of matrix
- Updated lattice semantics: bottom is identity, unknown is absorbing top
- Changed `Env.get()` to return `Shape.bottom()` for unbound variables
- Added bottom→unknown conversion in `eval_expr_ir` Var case (keeps bottom contained to Env level)
- Added defensive bottom handling in analysis_core.py and matrix_literals.py
- **Precision regression**: `join_shape(unknown, X)` now returns `unknown` (was `X`) — more sound but less precise
- 6 new test files (all in tests/loops/ and tests/control_flow/) + 2 updated tests
- Test count: 58 total (was 52)
- **Documentation locations to check**:
  1. CHANGELOG.md — Need [0.9.3] entry with Added/Changed/Fixed sections
  2. CLAUDE.md line 103 — Shape constructors (add `bottom`)
  3. CLAUDE.md line 114-120 — Shape System section (mention bottom vs unknown distinction)
  4. README.md line 53-60 — Shape System section (mention bottom if appropriate, but keep it user-friendly)
  5. README.md test table — 6 new tests to document (5 loops, 1 control_flow)
- **Key insight**: Lattice-level changes (Shape domain) require updating Shape System documentation in BOTH CLAUDE.md and README.md
- **Pattern**: When adding new Shape constructors, check Shape System sections in docs

### v0.10.0 User-Defined Functions (Phase C)
- Added support for user-defined functions in 3 syntactic forms (single return, multi-return, procedure)
- Interprocedural shape inference (call-site specific analysis)
- Dimension aliasing across function boundaries (symbolic names propagate from caller to callee)
- Destructuring assignment for multiple return values: `[a, b] = func(x)`
- New IR nodes: FunctionDef, AssignMulti
- New dataclasses: AnalysisContext (threads function registry, recursion guard, fixpoint flag), FunctionSignature (function metadata)
- Two-pass program analysis (Pass 1: register functions, Pass 2: analyze script)
- Recursion guard to prevent infinite loops (emits W_RECURSIVE_FUNCTION)
- Dual-location warnings (call site + body location)
- 6 new warning codes: W_FUNCTION_ARG_COUNT_MISMATCH, W_RECURSIVE_FUNCTION, W_PROCEDURE_IN_EXPR, W_MULTI_ASSIGN_NON_CALL, W_MULTI_ASSIGN_BUILTIN, W_MULTI_ASSIGN_COUNT_MISMATCH
- Added 9th test category: tests/functions/ with 8 new test files
- Test count: 66 total (was 58)
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.10.0] entry with comprehensive Added/Changed sections
  2. CLAUDE.md line 139 — Test categories list (added functions/)
  3. CLAUDE.md Architecture section — Added AnalysisContext/FunctionSignature details
  4. CLAUDE.md Critical Implementation Details — New "User-Defined Functions" subsection
  5. CLAUDE.md Apply Node section — Updated disambiguation logic to include user-defined functions
  6. CLAUDE.md Known Behaviors — Added note about call-site specific analysis
  7. README.md Language Subset section — Added user-defined functions to subset description
  8. README.md Shape System section — Added dimension aliasing to capabilities list
  9. README.md Test table — Added 8 new function test entries
  10. README.md Limitations section — Removed "user-defined functions" from limitations (now supported)
  11. README.md Future Directions — Removed interprocedural analysis from future work (now implemented)
- **Pattern**: Major feature additions require:
  - New CHANGELOG version entry (not Unreleased)
  - Architecture section updates in CLAUDE.md
  - Critical Implementation Details subsection if new analysis phase
  - Test category list update (CLAUDE.md line 139)
  - Language subset update in README.md
  - Limitations section cleanup (remove implemented features)
  - Test table expansion with all new test files
- **Key insight**: When removing a limitation, also check Future Directions section for related items to remove
- **AGENTS.md**: No changes required — workflow descriptions are implementation-agnostic

### v0.10.1 Polymorphic Caching and Return Statement
- Added polymorphic caching for function analysis (keyed by func_name, arg_shapes)
- Warning replay mechanism: cached warnings replayed with current call-site line numbers
- Return statement support (keyword, IR node, EarlyReturn exception)
- Catch-at-boundary semantics: EarlyReturn caught in If, loops, and program (stops analysis without propagating)
- New warning code: W_RETURN_OUTSIDE_FUNCTION
- 5 new test files in tests/functions/ (3 caching tests, 2 return statement tests)
- Test count: 71 total (was 66)
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.10.1] entry with Added/Changed/Infrastructure sections
  2. CLAUDE.md line 90 — IR node list (added Return to statement nodes)
  3. CLAUDE.md line 96 — AnalysisContext description (added polymorphic cache, EarlyReturn exception)
  4. CLAUDE.md lines 170-177 — User-Defined Functions section (added caching, warning replay, return statement, catch-at-boundary)
  5. README.md line 46 — Language subset (added "return" to control flow list)
  6. README.md lines 130-134 — Test table (added 5 new function test entries)
- **Pattern**: Incremental feature enhancements (caching, control flow keywords) require:
  - CHANGELOG entry with detailed Added/Changed sections
  - AnalysisContext field updates in CLAUDE.md Architecture section
  - User-Defined Functions subsection updates if related to functions
  - IR node list updates if new statement/expression node added
  - Language subset updates in README.md if new keyword
  - Test table expansion for all new tests
- **Key insight**: When adding new IR nodes, update BOTH Architecture section (line 90) AND Critical Implementation Details
- **Caching documentation pattern**: Document cache key structure, replay mechanism, and performance implications
- **Exception-based control flow pattern**: Document exception type, where raised, where caught (boundary semantics)

### v0.10.2 Test Coverage Polish
- Pure test addition: 11 new test files for edge cases and integration scenarios (no code changes)
- Return statement edge cases: return_in_if.m, return_in_loop.m, early_return_multi_output.m
- Cache edge cases: cache_symbolic_args.m, cache_warning_replay.m, arg_count_mismatch_cached.m
- Integration tests: function_in_loop.m, nested_function_calls.m, procedure_with_return.m
- Extended 2 existing tests: return_statement.m (+ unreachable code), multiple_returns.m (+ single-output usage)
- Test count: 80 total (was 71)
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.10.2] entry with Added section (test-only changes)
  2. README.md lines 133-143 — Test table (added 11 new function test entries, updated 2 descriptions)
- **Pattern**: Test-only releases require:
  - CHANGELOG entry documenting test additions and extensions
  - README test table expansion (all new tests + updated descriptions for extended tests)
  - NO CLAUDE.md updates (architecture unchanged)
  - Test count verification via `python3 mmshape.py --tests`
- **Key insight**: Test coverage polish passes are valuable documentation milestones — capture what edge cases are now tested

### v0.11.0 Extended Control Flow Constructs
- Added 4 new control flow constructs: elseif chains, break/continue, switch/case, try/catch
- 8 new parser keywords: elseif, break, continue, switch, case, otherwise, try, catch
- 5 new IR nodes: IfChain (replaces If), Switch, Try, Break, Continue
- Exception handling: EarlyBreak and EarlyContinue exceptions (caught in loop handlers)
- Full exception handling in all branch handlers (IfChain, Switch, Try)
- 12 new test files in tests/control_flow/ (3 elseif, 3 break/continue, 3 switch, 3 try)
- Test count: 92 total (was 80)
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.11.0] entry with Added/Changed sections
  2. CLAUDE.md line 90 — IR node list (replaced If with IfChain, added Switch, Try, Break, Continue)
  3. CLAUDE.md line 99 — Added EarlyBreak/EarlyContinue exception documentation
  4. README.md line 46 — Language subset (expanded control flow to list all constructs)
  5. README.md lines 114-129 — Test table (added 12 new control_flow test entries)
- **Pattern**: Control flow expansion requires:
  - CHANGELOG entry with keyword list, IR node changes, exception types
  - IR node list update in CLAUDE.md Architecture section (line 90)
  - Exception documentation in analysis_ir.py subsection (line 99)
  - Language subset update in README.md to list all control flow constructs explicitly
  - Test table expansion (all new test files with descriptive names)
  - Test count verification via `python3 mmshape.py --tests`
- **Key insight**: When control flow constructs are added, update language subset to list ALL constructs explicitly (not just "control flow") for completeness
- **Exception pattern**: Document exception types, what triggers them, and where they're caught (similar to EarlyReturn)
- **Note**: IfChain replaced If (breaking change in IR node naming), but If→IfChain is backward-compatible at syntax level (elseif is new keyword)

### v0.12.0 Language Extensions (Strings, Structs, Anonymous Functions)
- Added 3 new language features in 3 phases: strings (Phase 1), structs (Phase 2), anonymous functions (Phase 3)
- 3 new shape kinds: Shape.string(), Shape.struct(fields), Shape.function_handle()
- 5 new IR nodes: StringLit, FieldAccess, StructAssign, Lambda, FuncHandle
- 4 new warning codes: W_STRING_ARITHMETIC, W_STRUCT_FIELD_NOT_FOUND, W_FIELD_ACCESS_NON_STRUCT, W_LAMBDA_CALL_APPROXIMATE
- Context-sensitive lexer: `'` disambiguates string start vs transpose based on previous token
- Struct shape with sorted-tuple fields for hashability, union-with-bottom join semantics
- Lambda/function handle tracking in AnalysisContext with monotonic closure counter
- Apply disambiguation updated: function_handle variables shadow builtins
- 10th test category: tests/structs/ (5 files)
- 14 new tests total: 4 string (literals/), 5 struct (structs/), 5 lambda (functions/)
- Test count: 106 total (was 92)
- **Documentation locations updated**:
  1. CHANGELOG.md — New [0.12.0] entry with comprehensive Added/Changed sections (3-phase structure)
  2. CLAUDE.md line 91 — IR node list (added StringLit, FieldAccess, StructAssign, Lambda, FuncHandle to Expr; StructAssign to Stmt)
  3. CLAUDE.md lines 109-111 — Shape domain (added string, struct, function_handle kinds)
  4. CLAUDE.md line 118-127 — Shape System section (expanded to list 7 shape kinds explicitly)
  5. CLAUDE.md line 145 — Test categories (9→10, added structs/)
  6. README.md line 46 — Language subset (added strings, structs, anonymous functions)
  7. README.md line 53-62 — Shape System (expanded to list all 7 shape kinds)
  8. README.md test table — Added 14 new test entries (4 literals, 5 structs, 5 functions)
  9. README.md line 210-215 — Limitations (removed "cell arrays or structs", "strings", "anonymous functions")
- **Pattern**: Multi-phase feature releases require:
  - CHANGELOG entry organized by phase (Added/Changed per phase OR comprehensive Added/Changed covering all phases)
  - Shape domain updates in BOTH CLAUDE.md and README.md (list ALL shape kinds explicitly)
  - IR node list expansion (group by Expr vs Stmt)
  - Test category count update if new directory added
  - Language subset update to list new features
  - Limitations cleanup (remove ALL newly-supported features)
  - Test table expansion for all new tests (organized by category)
  - Warning code documentation in CHANGELOG
- **Key insight**: Shape kind additions are HIGH-IMPACT changes — update Shape System documentation in 3 locations:
  1. CLAUDE.md runtime/shapes.py section (line 109)
  2. CLAUDE.md Shape System subsection (line 118)
  3. README.md Shape System section (line 53)
- **Struct join semantics pattern**: When Shape.struct() uses union-with-bottom join (missing fields→bottom), document in CHANGELOG and code comments (not necessarily README)
- **Context-sensitive lexer pattern**: Major parser changes (like string/transpose disambiguation) belong in CHANGELOG Added section with technical detail
- **Apply disambiguation update**: When disambiguation logic changes (e.g., handle shadows builtins), update CLAUDE.md Apply Node section
- **Note**: Promoted tests/recovery/struct_field.m from W_UNSUPPORTED_STMT to W_FIELD_ACCESS_NON_STRUCT (recovery→first-class support)
