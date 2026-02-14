# Task: Extended Control Flow Constructs (v0.11.0)

## Goal
Implement four related control flow constructs missing from Mini-MATLAB: elseif chains, break/continue loop control, switch/case, and try/catch error handling.

## Scope
- **Parser**: Add keywords (elseif, break, continue, switch, case, otherwise, try, catch), parsing rules
- **IR**: Add nodes (IfChain, Break, Continue, Switch, Try)
- **Lowering**: Convert syntax AST to IR for new constructs
- **Analyzer**: Implement control flow semantics with appropriate environment joins
- **Tests**: 12 new test files (3 per feature) covering basic usage, edge cases, and interactions

## Non-goals
- No optimization or dead code elimination (e.g., unreachable case branches)
- No validation of switch case values (constant folding, duplicate detection)
- No analysis of catch block error variables (MATLAB's MException objects)
- No break/continue validation outside loop (EarlyBreak/EarlyContinue caught at top level gracefully)
- Keywords `case`, `catch`, `otherwise`, `break`, `continue` are reserved (cannot be variable names)

## Invariants Impacted
- **All tests pass**: 80 existing + 12 new = 92 total
- **IR analyzer authoritative**: Preserved (analyzer implements semantics)
- **Best-effort analysis**: All constructs continue analysis after warnings
- **Test format**: Standard `% EXPECT:` inline assertions

## Acceptance Criteria
- [ ] All 92 tests pass: `python3 mmshape.py --tests`
- [ ] All 92 tests pass in fixpoint mode: `python3 mmshape.py --fixpoint --tests`
- [ ] Parser accepts all four constructs without syntax errors
- [ ] Analyzer produces correct environment joins for elseif chains (multi-way join)
- [ ] Analyzer handles break/continue with 3-phase widening loop analysis correctly
- [ ] Switch/case joins all case environments (no fall-through)
- [ ] Try/catch uses conservative join (try-env + catch-env)
- [ ] New warning codes stable (W_BREAK_OUTSIDE_LOOP, W_CONTINUE_OUTSIDE_LOOP)

## Commands to Run
```bash
python3 mmshape.py --tests
python3 mmshape.py --fixpoint --tests
python3 mmshape.py tests/control_flow/elseif_chain.m
python3 mmshape.py tests/control_flow/elseif_no_else.m
python3 mmshape.py tests/control_flow/elseif_nested.m
python3 mmshape.py tests/control_flow/break_simple.m
python3 mmshape.py tests/control_flow/continue_simple.m
python3 mmshape.py tests/control_flow/break_nested_loop.m
python3 mmshape.py tests/control_flow/switch_basic.m
python3 mmshape.py tests/control_flow/switch_no_otherwise.m
python3 mmshape.py tests/control_flow/switch_mismatch.m
python3 mmshape.py tests/control_flow/try_catch_basic.m
python3 mmshape.py tests/control_flow/try_catch_no_error.m
python3 mmshape.py tests/control_flow/try_nested.m
```

## Design Notes

### 1. IfChain: Nested If vs First-Class Node
**Decision**: Use first-class `IfChain` IR node to preserve source fidelity and simplify analyzer logic.

**Lowering approach**:
```python
# Syntax: ['if', cond1, then1, elseifs, else_body]
#   where elseifs = [[cond2, body2], [cond3, body3], ...]
# IR: IfChain(line, conditions, bodies, else_body)
#   conditions: List[Expr]  # [cond1, cond2, cond3, ...]
#   bodies: List[List[Stmt]]  # [then1, then2, then3, ...]
#   else_body: List[Stmt]
```

**Analyzer semantics**:
```python
# Evaluate all conditions (side effects)
# Copy env for each branch
# Analyze each body independently
# Join all branch environments (N-way join)
for cond in conditions:
    eval_expr_ir(cond, ...)
branch_envs = [env.copy() for _ in (bodies + [else_body])]
for body, branch_env in zip(bodies + [else_body], branch_envs):
    for stmt in body:
        analyze_stmt_ir(stmt, branch_env, ...)
# Multi-way join using iterative join_env
result = branch_envs[0]
for other in branch_envs[1:]:
    result = join_env(result, other)
env.bindings = result.bindings
```

### 2. Break/Continue: Exception-Based Control Flow
**Decision**: Use `EarlyBreak` and `EarlyContinue` exceptions (like `EarlyReturn`) to unwind loop analysis.

**IR nodes**:
```python
@dataclass(frozen=True)
class Break(Stmt):
    """Break statement (exit enclosing loop)."""
    pass

@dataclass(frozen=True)
class Continue(Stmt):
    """Continue statement (skip to next iteration)."""
    pass
```

**Analyzer semantics**:
- `Break` raises `EarlyBreak` exception
- `Continue` raises `EarlyContinue` exception
- `_analyze_loop_body` catches both at boundary (doesn't propagate)
- For **default mode**: break/continue stops current iteration, result is env at break/continue point
- For **fixpoint mode** (3-phase widening):
  - Phase 1 (Discover): If break/continue occurs, use env-at-break as post-iteration env
  - Phase 2 (Stabilize): Re-analyze with widened env; break/continue again stops iteration
  - Phase 3 (Post-loop join): Widen pre-loop and post-break/continue envs (models partial execution)

**Top-level catch**: `analyze_program_ir` must catch `EarlyBreak` and `EarlyContinue` alongside `EarlyReturn` at script level so break/continue outside loops degrades gracefully instead of crashing.

### 3. Switch/Case: Multi-Way Branch (No Fall-Through)
**Decision**: MATLAB switch semantics differ from C — each case executes independently, no fall-through.

**IR node**:
```python
@dataclass(frozen=True)
class Switch(Stmt):
    """Switch/case statement.

    MATLAB switch does not fall through (unlike C).
    Each case is an independent branch.
    """
    expr: Expr  # Switch expression
    cases: List[Tuple[Expr, List[Stmt]]]  # [(case_val1, body1), ...]
    otherwise: List[Stmt]  # Otherwise block (may be empty)
```

**Analyzer semantics**:
```python
# Evaluate switch expression for side effects
switch_shape = eval_expr_ir(stmt.expr, env, ...)
# Evaluate all case expressions for side effects (even if unreachable)
for case_val, _ in stmt.cases:
    eval_expr_ir(case_val, env, ...)
# Analyze each case body + otherwise independently
branch_envs = []
for _, case_body in stmt.cases:
    branch_env = env.copy()
    for s in case_body:
        analyze_stmt_ir(s, branch_env, ...)
    branch_envs.append(branch_env)
# Otherwise branch
otherwise_env = env.copy()
for s in stmt.otherwise:
    analyze_stmt_ir(s, otherwise_env, ...)
branch_envs.append(otherwise_env)
# Multi-way join (same as elseif)
result = branch_envs[0]
for other in branch_envs[1:]:
    result = join_env(result, other)
env.bindings = result.bindings
```

**Non-goal**: No validation of case value types or detection of unreachable cases.

### 4. Try/Catch: Conservative Join
**Decision**: For shape analysis, catch block sees environment where any statement in try may have failed. Conservative approach: catch starts with pre-try env.

**IR node**:
```python
@dataclass(frozen=True)
class Try(Stmt):
    """Try/catch error handling.

    For shape analysis: catch block models "any statement in try may fail".
    Conservative: catch starts with pre-try environment.
    """
    try_body: List[Stmt]
    catch_body: List[Stmt]
    # Note: MATLAB catch can bind error to variable (catch err), but we ignore it
```

**Analyzer semantics**:
```python
# Analyze try block
pre_try_env = env.copy()
try_env = env.copy()
for stmt in stmt.try_body:
    analyze_stmt_ir(stmt, try_env, ...)
# Analyze catch block starting from pre-try env
# (models: error could occur at any point in try)
catch_env = pre_try_env.copy()
for stmt in stmt.catch_body:
    analyze_stmt_ir(stmt, catch_env, ...)
# Join try and catch outcomes
result = join_env(try_env, catch_env)
env.bindings = result.bindings
```

**Non-goal**: No analysis of MATLAB's MException object (error variable in `catch err`).

## Parser Changes

### Keywords Addition
```python
# frontend/matlab_parser.py line 17
KEYWORDS = {
    "for", "while", "if", "else", "elseif", "end",
    "switch", "case", "otherwise",
    "try", "catch",
    "break", "continue",
    "function", "return"
}
```

### Parsing Rules

**IfChain** (modify `parse_if`):
```python
def parse_if(self) -> Any:
    """Internal: ['if', cond, then_body, elseifs, else_body]
    where elseifs = [[cond2, body2], [cond3, body3], ...]
    """
    self.eat("IF")
    cond = self.parse_expr()
    then_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))

    elseifs = []
    while self.current().kind == "ELSEIF":
        self.eat("ELSEIF")
        elif_cond = self.parse_expr()
        elif_body = self.parse_block(until_kinds=("ELSE", "ELSEIF", "END"))
        elseifs.append([elif_cond, elif_body])

    else_body = [["skip"]]
    if self.current().kind == "ELSE":
        self.eat("ELSE")
        else_body = self.parse_block(until_kinds=("END",))

    self.eat("END")
    return ["if", cond, then_body, elseifs, else_body]
```

**Switch** (new):
```python
def parse_switch(self) -> Any:
    """Internal: ['switch', expr, cases, otherwise_body]
    where cases = [[case_val1, body1], [case_val2, body2], ...]
    """
    switch_tok = self.eat("SWITCH")
    expr = self.parse_expr()

    cases = []
    while self.current().kind == "CASE":
        self.eat("CASE")
        case_val = self.parse_expr()
        case_body = self.parse_block(until_kinds=("CASE", "OTHERWISE", "END"))
        cases.append([case_val, case_body])

    otherwise_body = [["skip"]]
    if self.current().kind == "OTHERWISE":
        self.eat("OTHERWISE")
        otherwise_body = self.parse_block(until_kinds=("END",))

    self.eat("END")
    return ["switch", expr, cases, otherwise_body]
```

**Try** (new):
```python
def parse_try(self) -> Any:
    """Internal: ['try', try_body, catch_body]
    Note: Ignores optional catch variable (catch err)
    """
    self.eat("TRY")
    try_body = self.parse_block(until_kinds=("CATCH", "END"))

    catch_body = [["skip"]]
    if self.current().kind == "CATCH":
        self.eat("CATCH")
        # Skip optional error variable
        if self.current().kind == "ID":
            self.eat("ID")
        catch_body = self.parse_block(until_kinds=("END",))

    self.eat("END")
    return ["try", try_body, catch_body]
```

**Break/Continue** (modify `parse_stmt`):
```python
# In parse_stmt, add:
elif tok.kind == "BREAK":
    tok = self.eat("BREAK")
    return ["break", tok.line]
elif tok.kind == "CONTINUE":
    tok = self.eat("CONTINUE")
    return ["continue", tok.line]
elif tok.kind == "SWITCH":
    return self.parse_switch()
elif tok.kind == "TRY":
    return self.parse_try()
```

**Recovery**: Update `recover_to_stmt_boundary` to add `CASE`, `OTHERWISE`, `CATCH` to the set of block-ending keywords (alongside existing `END`, `ELSE`, `ELSEIF`). Without this, parse error recovery inside a switch case could eat into the next case.

## IR Node Definitions

```python
# ir/ir.py — Add to Stmt subclasses

@dataclass(frozen=True)
class IfChain(Stmt):
    """If-elseif-else chain.

    Represents: if c1 ... elseif c2 ... elseif c3 ... else ... end
    conditions[0] is the if condition, rest are elseif conditions.
    bodies[0] is the then body, rest are elseif bodies.
    """
    conditions: List[Expr]
    bodies: List[List[Stmt]]
    else_body: List[Stmt]

@dataclass(frozen=True)
class Switch(Stmt):
    """Switch/case statement (MATLAB semantics: no fall-through)."""
    expr: Expr
    cases: List[Tuple[Expr, List[Stmt]]]  # [(case_val, body), ...]
    otherwise: List[Stmt]

@dataclass(frozen=True)
class Try(Stmt):
    """Try/catch error handling."""
    try_body: List[Stmt]
    catch_body: List[Stmt]

@dataclass(frozen=True)
class Break(Stmt):
    """Break statement (exit loop)."""
    pass

@dataclass(frozen=True)
class Continue(Stmt):
    """Continue statement (skip to next iteration)."""
    pass
```

## Lowering Changes

```python
# frontend/lower_ir.py — Add to lower_stmt

if tag == "if":
    # Updated: handle elseifs
    cond = lower_expr(stmt[1])
    then_body = [lower_stmt(x) for x in stmt[2]]
    elseifs = stmt[3]  # [[cond2, body2], ...]
    else_body = [lower_stmt(x) for x in stmt[4]]

    if not elseifs:
        # Simple if/else (backward compatible)
        return If(line=cond.line, cond=cond, then_body=then_body, else_body=else_body)
    else:
        # IfChain chain
        conditions = [cond] + [lower_expr(ec) for ec, _ in elseifs]
        bodies = [then_body] + [[lower_stmt(s) for s in body] for _, body in elseifs]
        return IfChain(line=cond.line, conditions=conditions, bodies=bodies, else_body=else_body)

if tag == "switch":
    # ['switch', expr, cases, otherwise_body]
    expr = lower_expr(stmt[1])
    cases = [(lower_expr(case_val), [lower_stmt(s) for s in case_body])
             for case_val, case_body in stmt[2]]
    otherwise = [lower_stmt(s) for s in stmt[3]]
    return Switch(line=expr.line, expr=expr, cases=cases, otherwise=otherwise)

if tag == "try":
    # ['try', try_body, catch_body]
    line = stmt[1][0][1] if stmt[1] else 0  # Line from first stmt in try_body
    try_body = [lower_stmt(s) for s in stmt[1]]
    catch_body = [lower_stmt(s) for s in stmt[2]]
    return Try(line=line, try_body=try_body, catch_body=catch_body)

if tag == "break":
    return Break(line=stmt[1])

if tag == "continue":
    return Continue(line=stmt[1])
```

## Analyzer Changes

### Exception Definitions
```python
# analysis/analysis_ir.py — Add after EarlyReturn

class EarlyBreak(Exception):
    """Raised by break statement to exit loop."""
    pass

class EarlyContinue(Exception):
    """Raised by continue statement to skip to next iteration."""
    pass
```

### Statement Handlers

**IfChain** (with full EarlyReturn/EarlyBreak/EarlyContinue handling):
```python
if isinstance(stmt, IfChain):
    # Evaluate all conditions for side effects
    for cond in stmt.conditions:
        _ = eval_expr_ir(cond, env, warnings, ctx)

    # Analyze all branches, tracking which ones returned/broke/continued
    all_bodies = list(stmt.bodies) + [stmt.else_body]
    branch_envs = []
    returned_flags = []
    deferred_exception = None  # EarlyBreak or EarlyContinue to re-raise

    for body in all_bodies:
        branch_env = env.copy()
        returned = False
        try:
            for s in body:
                analyze_stmt_ir(s, branch_env, warnings, ctx)
        except EarlyReturn:
            returned = True
        except (EarlyBreak, EarlyContinue) as exc:
            # Break/continue inside if inside loop: record and re-raise after join
            returned = True  # Exclude from join (same as EarlyReturn)
            deferred_exception = exc
        branch_envs.append(branch_env)
        returned_flags.append(returned)

    # If ALL branches returned/broke, propagate
    if all(returned_flags):
        if deferred_exception:
            raise type(deferred_exception)()
        raise EarlyReturn()

    # Join only non-returned branches
    live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
    result = live_envs[0]
    for other in live_envs[1:]:
        result = join_env(result, other)
    env.bindings = result.bindings
    return env
```

**Switch** (same EarlyReturn/EarlyBreak/EarlyContinue pattern as IfChain):
```python
if isinstance(stmt, Switch):
    _ = eval_expr_ir(stmt.expr, env, warnings, ctx)
    for case_val, _ in stmt.cases:
        _ = eval_expr_ir(case_val, env, warnings, ctx)

    all_bodies = [body for _, body in stmt.cases] + [stmt.otherwise]
    branch_envs = []
    returned_flags = []
    deferred_exception = None

    for body in all_bodies:
        branch_env = env.copy()
        returned = False
        try:
            for s in body:
                analyze_stmt_ir(s, branch_env, warnings, ctx)
        except EarlyReturn:
            returned = True
        except (EarlyBreak, EarlyContinue) as exc:
            returned = True
            deferred_exception = exc
        branch_envs.append(branch_env)
        returned_flags.append(returned)

    if all(returned_flags):
        if deferred_exception:
            raise type(deferred_exception)()
        raise EarlyReturn()

    live_envs = [e for e, r in zip(branch_envs, returned_flags) if not r]
    result = live_envs[0]
    for other in live_envs[1:]:
        result = join_env(result, other)
    env.bindings = result.bindings
    return env
```

**Try** (catches EarlyReturn/EarlyBreak/EarlyContinue per-block):
```python
if isinstance(stmt, Try):
    pre_try_env = env.copy()

    # Analyze try block
    try_env = env.copy()
    try_returned = False
    deferred_exception = None
    try:
        for s in stmt.try_body:
            analyze_stmt_ir(s, try_env, warnings, ctx)
    except EarlyReturn:
        try_returned = True
    except (EarlyBreak, EarlyContinue) as exc:
        try_returned = True
        deferred_exception = exc

    # Analyze catch block (starts from pre-try state)
    catch_env = pre_try_env.copy()
    catch_returned = False
    try:
        for s in stmt.catch_body:
            analyze_stmt_ir(s, catch_env, warnings, ctx)
    except EarlyReturn:
        catch_returned = True
    except (EarlyBreak, EarlyContinue) as exc:
        catch_returned = True
        if not deferred_exception:
            deferred_exception = exc

    # Propagation logic (same as If handler)
    if try_returned and catch_returned:
        if deferred_exception:
            raise type(deferred_exception)()
        raise EarlyReturn()
    elif try_returned:
        env.bindings = catch_env.bindings
    elif catch_returned:
        env.bindings = try_env.bindings
    else:
        result = join_env(try_env, catch_env)
        env.bindings = result.bindings
    return env
```

**Break/Continue**:
```python
if isinstance(stmt, Break):
    # For v0.11.0: raise exception unconditionally
    # v0.11.1 can add loop_depth validation
    raise EarlyBreak()

if isinstance(stmt, Continue):
    raise EarlyContinue()
```

### Loop Handler Update

Modify `_analyze_loop_body` to catch `EarlyBreak` and `EarlyContinue`:

```python
def _analyze_loop_body(body: list, env: Env, warnings: List[str], ctx: AnalysisContext) -> None:
    """Analyze a loop body, handling break/continue and optionally using widening.

    Catches EarlyReturn, EarlyBreak, EarlyContinue at boundary (doesn't propagate).
    """
    if not ctx.fixpoint:
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except (EarlyReturn, EarlyBreak, EarlyContinue):
            pass  # Stop iteration, don't propagate
        return

    # Phase 1 (Discover)
    pre_loop_env = env.copy()
    try:
        for s in body:
            analyze_stmt_ir(s, env, warnings, ctx)
    except (EarlyReturn, EarlyBreak, EarlyContinue):
        pass  # Phase 1 stopped early

    # Widen
    widened = widen_env(pre_loop_env, env)

    # Phase 2 (Stabilize)
    if widened.bindings != env.bindings:
        env.bindings = widened.bindings
        try:
            for s in body:
                analyze_stmt_ir(s, env, warnings, ctx)
        except (EarlyReturn, EarlyBreak, EarlyContinue):
            pass  # Phase 2 stopped early

    # Phase 3 (Post-loop join)
    final = widen_env(pre_loop_env, env)
    env.bindings = final.bindings
```

### Warning Functions

```python
# analysis/diagnostics.py — Add new warnings

def warn_break_outside_loop(line: int) -> str:
    """Warning for break statement outside loop (v0.11.1 will validate)."""
    return f"W_BREAK_OUTSIDE_LOOP line {line}: break statement outside loop (treated as no-op)"

def warn_continue_outside_loop(line: int) -> str:
    """Warning for continue statement outside loop (v0.11.1 will validate)."""
    return f"W_CONTINUE_OUTSIDE_LOOP line {line}: continue statement outside loop (treated as no-op)"
```

Note: For v0.11.0, break/continue outside loops will raise exceptions at script level and be caught by the top-level try/except. v0.11.1 can add loop_depth tracking to emit these warnings gracefully.

## Tests to Add

### IfChain Tests (3 files)

**1. tests/control_flow/elseif_chain.m**
```matlab
% Test: If-elseif-else chain with compatible branches
% All branches assign matrix[3 x 3], join preserves shape
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]

n = 5;
if n < 3
    A = zeros(3, 3);
elseif n < 6
    A = ones(3, 3);
elseif n < 10
    A = eye(3);
else
    A = rand(3, 3);
end
```

**2. tests/control_flow/elseif_no_else.m**
```matlab
% Test: Elseif chain without else clause
% Missing else branch means A may be uninitialized (bottom → unknown via join)
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

n = 2;
if n > 10
    A = zeros(4, 4);
elseif n > 5
    A = ones(4, 4);
end
```

**3. tests/control_flow/elseif_mismatch.m**
```matlab
% Test: Elseif branches with shape mismatch
% Different branch shapes → join produces less precise result
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x None]

k = 4;
if k == 1
    B = zeros(3, 4);
elseif k == 2
    B = zeros(3, 5);
else
    B = zeros(3, 6);
end
```

### Break/Continue Tests (3 files)

**4. tests/control_flow/break_simple.m**
```matlab
% Test: Break statement in for loop
% Loop body may not complete all iterations
% EXPECT: warnings = 0
% EXPECT: A = matrix[10 x 10]
% EXPECT: i = scalar

A = zeros(10, 10);
for i = 1:100
    if i > 10
        break;
    end
    A = eye(10);
end
```

**5. tests/control_flow/continue_simple.m**
```matlab
% Test: Continue statement in while loop
% Continue skips to next iteration, doesn't affect shape
% EXPECT: warnings = 0
% EXPECT: B = matrix[5 x 5]
% EXPECT: count = scalar

count = 0;
B = zeros(5, 5);
while count < 20
    count = count + 1;
    if count < 10
        continue;
    end
    B = ones(5, 5);
end
```

**6. tests/control_flow/break_nested_loop.m**
```matlab
% Test: Break in nested loop (only exits inner loop)
% Outer loop continues after inner break
% EXPECT: warnings = 0
% EXPECT: C = matrix[3 x 3]
% EXPECT: i = scalar
% EXPECT: j = scalar

C = zeros(3, 3);
for i = 1:5
    for j = 1:10
        if j > 3
            break;
        end
        C = eye(3);
    end
end
```

### Switch/Case Tests (3 files)

**7. tests/control_flow/switch_basic.m**
```matlab
% Test: Switch/case with otherwise
% All branches assign compatible shapes
% EXPECT: warnings = 0
% EXPECT: result = matrix[2 x 2]

mode = 1;
switch mode
    case 1
        result = zeros(2, 2);
    case 2
        result = ones(2, 2);
    otherwise
        result = eye(2);
end
```

**8. tests/control_flow/switch_no_otherwise.m**
```matlab
% Test: Switch without otherwise clause
% Missing otherwise means result may be uninitialized
% EXPECT: warnings = 0
% EXPECT: output = matrix[4 x 4]

val = 10;
switch val
    case 1
        output = zeros(4, 4);
    case 2
        output = ones(4, 4);
end
```

**9. tests/control_flow/switch_mismatch.m**
```matlab
% Test: Switch cases with shape conflicts
% Different case shapes → join loses precision
% EXPECT: warnings = 0
% EXPECT: M = matrix[None x None]

choice = 3;
switch choice
    case 1
        M = zeros(3, 3);
    case 2
        M = zeros(4, 4);
    case 3
        M = zeros(5, 5);
    otherwise
        M = zeros(6, 6);
end
```

### Try/Catch Tests (3 files)

**10. tests/control_flow/try_catch_basic.m**
```matlab
% Test: Try/catch with error in try block
% A is 3x4, B is 5x3 → inner dim mismatch (4 != 5), X = unknown in try
% Catch starts from pre-try env, assigns X = matrix[1 x 1]
% Join: unknown (try) + matrix[1 x 1] (catch) = unknown
% EXPECT: warnings = 1
% EXPECT: X = unknown

try
    A = zeros(3, 4);
    B = zeros(5, 3);
    X = A * B;
catch
    X = zeros(1, 1);
end
```

**11. tests/control_flow/try_catch_no_error.m**
```matlab
% Test: Try/catch with no error
% Both try and catch branches analyzed, joined
% EXPECT: warnings = 0
% EXPECT: Y = matrix[5 x 5]

try
    Y = zeros(5, 5);
catch
    Y = ones(5, 5);
end
```

**12. tests/control_flow/try_nested.m**
```matlab
% Test: Nested try/catch blocks
% Inner try/catch joins, outer try/catch joins result
% EXPECT: warnings = 0
% EXPECT: Z = matrix[2 x 2]

try
    try
        Z = zeros(2, 2);
    catch
        Z = eye(2);
    end
catch
    Z = ones(2, 2);
end
```

## Risks & Mitigations

**Risk**: Break/continue outside loop causes unhandled exception at script level.
**Mitigation**: For v0.11.0, exceptions propagate naturally (analysis stops). v0.11.1 can add loop_depth tracking to emit warnings gracefully.

**Risk**: Multi-way join (elseif, switch) is O(N) sequential joins; large switch may be slow.
**Mitigation**: Accept for v0.11.0 (typical switches have <10 cases). Future optimization can implement associative join tree.

**Risk**: Try/catch conservative join loses precision when try block has no errors.
**Mitigation**: Acceptable for soundness. More precise analysis would require exception-flow tracking (out of scope).

**Risk**: IfChain first-class node duplicates some If logic.
**Mitigation**: Clean separation of concerns. IfChain handler is ~20 lines, manageable.

## Estimated Diff

- **Parser** (~80 lines): 8 keywords, 3 parsing methods (parse_switch, parse_try, modify parse_if), stmt dispatch
- **IR** (~25 lines): 5 new dataclass nodes (IfChain, Switch, Try, Break, Continue)
- **Lowering** (~50 lines): 5 new lowering cases (if with elseifs, switch, try, break, continue)
- **Analyzer** (~100 lines): 5 new statement handlers, 2 exception classes, loop handler update
- **Diagnostics** (~10 lines): 2 new warning functions (break/continue outside loop — deferred)
- **Tests** (~180 lines): 12 new test files (avg 15 lines each)
- **Total**: ~445 lines of code + tests

**Modified files**: 6 core files (parser, IR, lowering, analyzer, diagnostics, imports)
**New test files**: 12 in tests/control_flow/
**Test count**: 80 → 92 (+12)
