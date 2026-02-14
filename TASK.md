# Task: User-Defined Function Support (v0.10.0)

## Goal
Add interprocedural shape inference for user-defined MATLAB functions. Enable parsing function declarations, tracking separate function workspaces, and propagating argument shapes from call sites to infer return value shapes. Use dimension aliasing to preserve symbolic dimension names across function boundaries.

## Scope
- Parse function declarations in all 3 forms: single return, multiple returns, procedure (no return)
- Parse destructuring assignment: `[a, b] = func(x)`
- Add IR nodes: `FunctionDef`, `AssignMulti`
- Add `dim_aliases: Dict[str, str]` to `Env` class for symbolic dimension propagation
- Thread `AnalysisContext` dataclass through analyzer (replaces global `_function_registry`)
- Two-pass program analysis: register functions, then analyze script code
- Demand-driven function analysis: re-analyze function body at every call site with caller's argument shapes
- Destructuring assignment binds multiple function outputs to multiple variables
- Support mixed files (functions + script code)
- Dual-location warnings for function body errors (show call site + body line)
- Recursion guard: emit `W_RECURSIVE_FUNCTION` and return unknown

## Non-goals
- Monomorphic caching (dropped — re-analyze every call)
- Varargs (`varargin`, `varargout`) — deferred
- Function handles or anonymous functions (`@(x) x^2`) — deferred
- Nested functions or subfunctions in separate files — deferred
- Explicit `return` statement (parser support deferred; implicit return from output vars is sufficient)
- Multi-file programs (each .m file is analyzed independently) — deferred
- Global variables (`global` keyword) — deferred
- Path-sensitive function analysis (always re-analyze, even if same shapes) — deferred to polymorphism work

## Invariants Impacted
- **IR analyzer authoritative**: Preserved (function analysis integrated into IR analyzer)
- **All tests pass**: New tests added, all existing tests continue passing
- **Minimal diffs**: Parser adds function syntax, analyzer adds function call handling, no changes to existing shape inference logic
- **Apply node unchanged**: User-defined function calls flow through existing Apply node disambiguation
- **Dimension alias correctness**: `expr_to_dim_ir(Var("n"))` in function body must return caller's dimension name, not function's parameter name

## Acceptance Criteria
- [ ] Parser recognizes `function` keyword and parses all 3 function forms (single return, multi return, procedure)
- [ ] Parser recognizes destructuring assignment `[a, b] = expr` (lookahead disambiguates from matrix literal)
- [ ] `FunctionDef` IR node stores function signature (name, params, output_vars, body)
- [ ] `AssignMulti` IR node stores multiple assignment targets
- [ ] `Env` class has `dim_aliases: Dict[str, str]` field (default empty dict)
- [ ] `AnalysisContext` dataclass threads through analyzer (contains function_registry, analyzing_functions, fixpoint)
- [ ] `analyze_program_ir` two-pass: register functions, analyze script statements
- [ ] `analyze_function_call` re-analyzes function body at every call site (no monomorphic cache)
- [ ] Dimension aliases propagate symbolic dimension names from caller to function body
- [ ] Destructuring assignment extracts multiple output shapes and binds to targets
- [ ] Function body warnings show both call site and body location
- [ ] Recursion guard prevents infinite loop (emit `W_RECURSIVE_FUNCTION`)
- [ ] Procedure-style functions (no return) emit warning if used in expression context
- [ ] Functions defined inside control flow are accepted silently (MATLAB forbids this, but analyzer accepts)
- [ ] Fixpoint flag inherited by function body analysis
- [ ] All existing tests pass: `python3 mmshape.py --tests`
- [ ] 7 new function tests pass (see Tests to Add)

## Commands to Run
```bash
python3 mmshape.py --tests
python3 mmshape.py tests/functions/simple_function.m
python3 mmshape.py tests/functions/multiple_returns.m
python3 mmshape.py tests/functions/matrix_constructor.m
python3 mmshape.py tests/functions/procedure.m
python3 mmshape.py tests/functions/unknown_in_function.m
python3 mmshape.py tests/functions/function_then_script.m
python3 mmshape.py tests/functions/call_with_mismatch.m
python3 mmshape.py tests/functions/recursion.m
```

## Tests to Add

### `tests/functions/simple_function.m`
Single-argument, single-return function.
```matlab
% Test: Simple single-argument, single-return function
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]

function y = square_matrix(x)
    y = x * x;
end

A = zeros(3, 3);
B = square_matrix(A);
```

### `tests/functions/multiple_returns.m`
Multiple return values via destructuring.
```matlab
% Test: Function with multiple return values
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[4 x 3]

function [out1, out2] = transpose_pair(in1)
    out1 = in1;
    out2 = in1';
end

[A, B] = transpose_pair(zeros(3, 4));
```

### `tests/functions/matrix_constructor.m`
Function that constructs matrix from dimension args (tests dimension aliasing).
```matlab
% Test: Function using dimension arguments to construct matrix
% Dimension aliasing: make_matrix(n, m) should return matrix[n x m], not matrix[rows x cols]
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x m]

function result = make_matrix(rows, cols)
    result = zeros(rows, cols);
end

A = make_matrix(n, m);
```

### `tests/functions/procedure.m`
Procedure-style function (no return value).
```matlab
% Test: Procedure-style function (no return value)
% EXPECT: warnings = 1
% EXPECT: A = unknown

function myproc(x)
    y = x * x;
end

A = myproc(zeros(3, 3));
```

### `tests/functions/unknown_in_function.m`
Function body calls unknown builtin (error propagates to caller).
```matlab
% Test: Function calls unknown builtin, propagates unknown to caller
% Dual-location warning: "line 3 (in bad_func, called from line 6)"
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = bad_func(x)
    y = unknown_builtin(x);
end

A = bad_func(zeros(3, 3));
```

### `tests/functions/function_then_script.m`
Mixed file: function definition followed by script code.
```matlab
% Test: Mixed file — function definition followed by script code
% EXPECT: warnings = 0
% EXPECT: X = matrix[5 x 5]
% EXPECT: Y = matrix[5 x 5]

function out = identity_wrapper(in)
    out = in;
end

X = zeros(5, 5);
Y = identity_wrapper(X);
```

### `tests/functions/call_with_mismatch.m`
Function call where argument shape causes mismatch in function body.
```matlab
% Test: Function called with incompatible argument shape
% Function body expects square matrix (x*x), caller passes non-square
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = unknown

function y = square_it(x)
    y = x * x;
end

A = zeros(3, 4);
B = square_it(A);
```

### `tests/functions/recursion.m`
Recursive function (guard emits warning and returns unknown).
```matlab
% Test: Recursive function (recursion guard prevents infinite loop)
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = factorial(n)
    y = n * factorial(n - 1);
end

A = factorial(5);
```

## Design Notes

### AnalysisContext Dataclass

Replaces global `_function_registry`. Threads through all analysis functions.

```python
@dataclass
class AnalysisContext:
    function_registry: Dict[str, FunctionSignature] = field(default_factory=dict)
    analyzing_functions: Set[str] = field(default_factory=set)
    fixpoint: bool = False
```

**FunctionSignature** (no result_shapes cache):
```python
@dataclass
class FunctionSignature:
    name: str
    params: List[str]
    output_vars: List[str]
    body: List[Stmt]
```

**Signature changes**:
- `analyze_program_ir(program, fixpoint=False)` → `analyze_program_ir(program, fixpoint=False, ctx=None)`
  - Creates `ctx = AnalysisContext(fixpoint=fixpoint)` if not provided
- `analyze_stmt_ir(stmt, env, warnings, fixpoint=False)` → `analyze_stmt_ir(stmt, env, warnings, ctx)`
- `eval_expr_ir(expr, env, warnings)` → `eval_expr_ir(expr, env, warnings, ctx)`
- `analyze_function_call(...)` takes `ctx` instead of accessing global state

### Dimension Aliasing in Env

Add `dim_aliases: Dict[str, str]` to `Env` class:

```python
@dataclass
class Env:
    bindings: Dict[str, Shape] = field(default_factory=dict)
    dim_aliases: Dict[str, str] = field(default_factory=dict)
```

**Alias setup at call site** (in `analyze_function_call`):
```python
func_env = Env()
for param_name, arg in zip(sig.params, args):
    arg_shape = _eval_index_arg_to_shape(arg, env, warnings, ctx)
    func_env.set(param_name, arg_shape)

    # Set up dimension alias if arg is a Var
    if isinstance(arg, IndexExpr) and isinstance(arg.expr, Var):
        caller_var = arg.expr.name
        caller_dim = expr_to_dim_ir(arg.expr, env)
        if caller_dim is not None:
            func_env.dim_aliases[param_name] = caller_dim
```

**Dimension extraction with aliasing** (modify `expr_to_dim_ir`):
```python
def expr_to_dim_ir(expr: Expr, env: Env) -> Dim:
    if isinstance(expr, Const):
        v = expr.value
        if float(v).is_integer():
            return int(v)
        return None
    if isinstance(expr, Var):
        # Check for dimension alias first
        if expr.name in env.dim_aliases:
            return env.dim_aliases[expr.name]
        return expr.name
    if isinstance(expr, BinOp):
        # ... existing BinOp handling ...
    return None
```

**Example**:
- Caller: `A = make_matrix(n, m)` where `n` and `m` are unbound variables
- Call site extracts: `expr_to_dim_ir(Var("n"), env) → "n"`, stores alias `"rows" → "n"`
- Function body: `zeros(rows, cols)` uses `expr_to_dim_ir(Var("rows"), func_env)` → looks up alias → returns `"n"`
- Result: `matrix[n x m]` not `matrix[rows x cols]`

### Parser Changes

**Add "function" keyword** (`frontend/matlab_parser.py` line 17):
```python
KEYWORDS = {"for", "while", "if", "else", "end", "function"}
```

**Add `parse_function()` method**:
```python
def parse_function(self) -> Any:
    """Parse function declaration.
    Internal: ['function', line, output_vars, name, params, body]

    Syntax:
      function result = name(arg1, arg2)           # single return
      function [out1, out2] = name(arg1, arg2)    # multiple returns
      function name(arg1, arg2)                   # procedure (no return)
    """
    func_tok = self.eat("FUNCTION")
    line = func_tok.line

    # Parse output variables (or none for procedure)
    output_vars = []

    # Check for procedure form: next token is ID followed by (
    if self.current().kind == "ID":
        # Peek ahead: is it "ID(" (procedure) or "ID =" (single return)?
        lookahead_tok = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        if lookahead_tok and lookahead_tok.value == "(":
            # Procedure form: function name(args)
            name = self.eat("ID").value
            self.eat("(")
        elif lookahead_tok and lookahead_tok.value == "=":
            # Single return: function result = name(args)
            output_vars.append(self.eat("ID").value)
            self.eat("=")
            name = self.eat("ID").value
            self.eat("(")
        else:
            raise ParseError(f"Expected '=' or '(' after function name at {self.current().pos}")
    elif self.current().value == "[":
        # Multiple outputs: function [a, b] = name(args)
        self.eat("[")
        output_vars.append(self.eat("ID").value)
        while self.current().value == ",":
            self.eat(",")
            output_vars.append(self.eat("ID").value)
        self.eat("]")
        self.eat("=")
        name = self.eat("ID").value
        self.eat("(")
    else:
        raise ParseError(f"Expected function output or name at {self.current().pos}")

    # Parse parameters
    params = []
    if self.current().value != ")":
        params.append(self.eat("ID").value)
        while self.current().value == ",":
            self.eat(",")
            params.append(self.eat("ID").value)
    self.eat(")")

    # Parse function body
    body = self.parse_block(until_kinds=("END",))
    self.eat("END")

    return ["function", line, output_vars, name, params, body]
```

**Modify `parse_program()` to handle functions**:
```python
def parse_program(self) -> Any:
    """Internal form: ['seq', stmt_or_func1, stmt_or_func2, ...]"""
    items = []
    while not self.at_end():
        # Skip newlines/semicolons
        while self.current().kind == "NEWLINE" or self.current().value == ";":
            if self.current().kind == "NEWLINE":
                self.eat("NEWLINE")
            else:
                self.eat(";")
            if self.at_end():
                return ["seq"] + items
        if self.at_end():
            break

        # Check for function definition
        if self.current().kind == "FUNCTION":
            items.append(self.parse_function())
        else:
            items.append(self.parse_stmt())

    return ["seq"] + items
```

**Add destructuring assignment parsing** (modify `parse_simple_stmt`):
```python
def parse_simple_stmt(self) -> Any:
    """Parse assignment or expression statement.

    Supports:
    - ID = expr
    - [ID, ID, ...] = expr  (destructuring assignment)
    - expr (expression statement)
    """
    if self.current().value == "[":
        # Check for destructuring assignment: [a, b] = expr
        # Lookahead to distinguish from matrix literal
        # Strategy: peek ahead to find ] = pattern
        saved_pos = self.i
        try:
            self.eat("[")
            # Parse ID list
            targets = [self.eat("ID").value]
            while self.current().value == ",":
                self.eat(",")
                targets.append(self.eat("ID").value)
            self.eat("]")

            # Check for =
            if self.current().value == "=":
                # Destructuring assignment confirmed
                eq_tok = self.eat("=")
                expr = self.parse_expr()
                return ["assign_multi", eq_tok.line, targets, expr]
            else:
                # Not destructuring, backtrack and parse as matrix literal
                self.i = saved_pos
                expr = self.parse_expr()
                return ["expr", expr]
        except ParseError:
            # Backtrack and parse as matrix literal
            self.i = saved_pos
            expr = self.parse_expr()
            return ["expr", expr]

    if self.current().kind == "ID":
        id_tok = self.eat("ID")
        if self.current().value == "=":
            self.eat("=")
            expr = self.parse_expr()
            return ["assign", id_tok.line, id_tok.value, expr]
        else:
            # Expression statement starting with ID
            expr_tail = self.parse_expr_rest(["var", id_tok.line, id_tok.value], 0)
            return ["expr", expr_tail]
    else:
        expr = self.parse_expr()
        return ["expr", expr]
```

### IR Changes

**Add `FunctionDef` statement** (`ir/ir.py`):
```python
@dataclass(frozen=True)
class FunctionDef(Stmt):
    """Function definition.

    Represents: function [out1, out2] = name(in1, in2)
    Or single return: function result = name(in1)
    Or procedure: function name(in1)
    """
    name: str
    params: List[str]  # Input parameter names
    output_vars: List[str]  # Output variable names (empty for procedures)
    body: List[Stmt]
```

**Add `AssignMulti` statement** (`ir/ir.py`):
```python
@dataclass(frozen=True)
class AssignMulti(Stmt):
    """Multiple assignment (destructuring).

    Represents: [a, b, c] = expr
    """
    targets: List[str]  # Variable names to assign
    expr: Expr  # Expression (must evaluate to multiple values)
```

### Lowering Changes

**Add `lower_function()`** (`frontend/lower_ir.py`):
```python
def lower_function(func: Any) -> FunctionDef:
    """Convert syntax function to IR FunctionDef.

    Args:
        func: ['function', line, output_vars, name, params, body]

    Returns:
        IR FunctionDef
    """
    tag, line, output_vars, name, params, body = func
    assert tag == "function"
    return FunctionDef(
        line=line,
        name=name,
        params=params,
        output_vars=output_vars,
        body=[lower_stmt(s) for s in body]
    )
```

**Modify `lower_program()`** (`frontend/lower_ir.py`):
```python
def lower_program(ast: Any) -> Program:
    """Convert syntax AST to IR Program.

    Args:
        ast: List-based syntax AST with ['seq', item1, item2, ...]
        Items can be statements or function definitions.

    Returns:
        IR Program
    """
    assert isinstance(ast, list) and ast and ast[0] == "seq"
    body = []
    for item in ast[1:]:
        if isinstance(item, list) and item[0] == "function":
            body.append(lower_function(item))
        else:
            body.append(lower_stmt(item))
    return Program(body=body)
```

**Add destructuring assignment lowering** (`frontend/lower_ir.py`):
```python
# In lower_stmt(), add case for assign_multi:
if tag == "assign_multi":
    # ['assign_multi', line, targets, expr]
    line = stmt[1]
    targets = stmt[2]
    expr = lower_expr(stmt[3])
    return AssignMulti(line=line, targets=targets, expr=expr)
```

### Analyzer Changes

**Modify `analyze_program_ir()`** (`analysis/analysis_ir.py`):
```python
def analyze_program_ir(program: Program, fixpoint: bool = False, ctx: AnalysisContext = None) -> Tuple[Env, List[str]]:
    """Analyze a complete Mini-MATLAB program for shape consistency.

    Two-pass analysis:
    1. Register all function definitions
    2. Analyze script statements (non-function statements in program body)

    Args:
        program: IR program to analyze
        fixpoint: If True, use fixed-point iteration for loop analysis
        ctx: Analysis context (created if not provided)

    Returns:
        Tuple of (final environment, list of warning messages)
    """
    if ctx is None:
        ctx = AnalysisContext(fixpoint=fixpoint)

    env = Env()
    warnings: List[str] = []

    # Pass 1: Register function definitions
    for item in program.body:
        if isinstance(item, FunctionDef):
            ctx.function_registry[item.name] = FunctionSignature(
                name=item.name,
                params=item.params,
                output_vars=item.output_vars,
                body=item.body
            )

    # Pass 2: Analyze script statements (non-functions)
    for item in program.body:
        if not isinstance(item, FunctionDef):
            analyze_stmt_ir(item, env, warnings, ctx)

    # Deduplicate warnings while preserving order
    warnings = list(dict.fromkeys(warnings))
    return env, warnings
```

**Add `analyze_function_call()`** (`analysis/analysis_ir.py`):
```python
def analyze_function_call(
    func_name: str,
    args: List[IndexArg],
    line: int,
    env: Env,
    warnings: List[str],
    ctx: AnalysisContext
) -> List[Shape]:
    """Analyze user-defined function call and return output shapes.

    Re-analyzes function body at every call site (no caching).

    Args:
        func_name: Name of function to call
        args: Argument list from Apply node
        line: Call site line number
        env: Caller's environment
        warnings: List to append warnings to
        ctx: Analysis context

    Returns:
        List of output shapes (one per output_var in function signature)
    """
    if func_name not in ctx.function_registry:
        # Should not reach here (checked by caller)
        return [Shape.unknown()]

    sig = ctx.function_registry[func_name]

    # Check argument count
    if len(args) != len(sig.params):
        warnings.append(diag.warn_function_arg_count_mismatch(
            line, func_name, expected=len(sig.params), got=len(args)
        ))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Recursion guard
    if func_name in ctx.analyzing_functions:
        warnings.append(diag.warn_recursive_function(line, func_name))
        return [Shape.unknown()] * max(len(sig.output_vars), 1)

    # Mark function as currently being analyzed
    ctx.analyzing_functions.add(func_name)

    try:
        # Analyze function body with fresh workspace
        func_env = Env()
        func_warnings: List[str] = []

        # Bind parameters to argument shapes + set up dimension aliases
        for param_name, arg in zip(sig.params, args):
            arg_shape = _eval_index_arg_to_shape(arg, env, warnings, ctx)
            func_env.set(param_name, arg_shape)

            # Dimension aliasing: if arg is a Var, extract its dimension
            if isinstance(arg, IndexExpr) and isinstance(arg.expr, Var):
                caller_dim = expr_to_dim_ir(arg.expr, env)
                if caller_dim is not None:
                    func_env.dim_aliases[param_name] = caller_dim

        # Analyze function body (inherit fixpoint setting)
        for stmt in sig.body:
            analyze_stmt_ir(stmt, func_env, func_warnings, ctx)

        # Prepend call context to function warnings (dual-location format)
        for func_warn in func_warnings:
            # Extract warning line from func_warn (format: "W_... line N: ...")
            if " line " in func_warn:
                parts = func_warn.split(" line ", 1)
                if len(parts) == 2:
                    prefix = parts[0]
                    rest = parts[1]
                    # rest = "N: message" or "N (in ..., called from ...): message"
                    if ": " in rest:
                        body_line = rest.split(": ", 1)[0].split(" ")[0]
                        message = rest.split(": ", 1)[1]
                        # Check if already has call context
                        if "(in " not in rest:
                            # Add call context
                            new_warn = f"{prefix} line {body_line} (in {func_name}, called from line {line}): {message}"
                            warnings.append(new_warn)
                        else:
                            # Already has call context, append as-is
                            warnings.append(func_warn)
                    else:
                        warnings.append(func_warn)
                else:
                    warnings.append(func_warn)
            else:
                warnings.append(func_warn)

        # Extract return values from output variables
        result_shapes = []
        for out_var in sig.output_vars:
            result_shapes.append(func_env.get(out_var))

        # Return output shapes (or single unknown if no outputs)
        return result_shapes if result_shapes else [Shape.unknown()]

    finally:
        # Remove function from analyzing set
        ctx.analyzing_functions.discard(func_name)
```

**Modify `eval_expr_ir()` Apply case** (line 391-393):
```python
# In Apply node handling, before emitting W_UNKNOWN_FUNCTION:
# (inside the fname not in env.bindings check)

# Check function registry before giving up
if fname in ctx.function_registry:
    # User-defined function call
    output_shapes = analyze_function_call(fname, expr.args, line, env, warnings, ctx)
    if len(output_shapes) == 1:
        return output_shapes[0]
    elif len(output_shapes) > 1:
        # Multiple returns in expression context → warning
        warnings.append(diag.warn_multiple_returns_in_expr(line, fname))
        return Shape.unknown()
    else:
        # Procedure (no return) in expression context → warning
        warnings.append(diag.warn_procedure_in_expr(line, fname))
        return Shape.unknown()

# Truly unknown function
warnings.append(diag.warn_unknown_function(line, fname))
return Shape.unknown()
```

**Add `AssignMulti` handling in `analyze_stmt_ir()`** (`analysis/analysis_ir.py`):
```python
if isinstance(stmt, AssignMulti):
    # Destructuring assignment: [a, b] = expr
    # Evaluate RHS (must be function call)
    if not isinstance(stmt.expr, Apply):
        warnings.append(diag.warn_multi_assign_non_call(stmt.line))
        for target in stmt.targets:
            env.set(target, Shape.unknown())
        return env

    # Extract function name
    if not isinstance(stmt.expr.base, Var):
        warnings.append(diag.warn_multi_assign_non_call(stmt.line))
        for target in stmt.targets:
            env.set(target, Shape.unknown())
        return env

    fname = stmt.expr.base.name

    # Check if builtin (builtins don't support multiple returns)
    if fname in KNOWN_BUILTINS:
        warnings.append(diag.warn_multi_assign_builtin(stmt.line, fname))
        for target in stmt.targets:
            env.set(target, Shape.unknown())
        return env

    # Check function registry
    if fname in ctx.function_registry:
        output_shapes = analyze_function_call(fname, stmt.expr.args, stmt.line, env, warnings, ctx)

        # Check target count matches output count
        if len(stmt.targets) != len(output_shapes):
            warnings.append(diag.warn_multi_assign_count_mismatch(
                stmt.line, fname, expected=len(output_shapes), got=len(stmt.targets)
            ))
            for target in stmt.targets:
                env.set(target, Shape.unknown())
        else:
            # Bind targets to outputs
            for target, shape in zip(stmt.targets, output_shapes):
                env.set(target, shape)
        return env

    # Not a known function
    warnings.append(diag.warn_unknown_function(stmt.line, fname))
    for target in stmt.targets:
        env.set(target, Shape.unknown())
    return env
```

**Thread `ctx` through helper functions**:
- `_eval_index_arg_to_shape(arg, env, warnings)` → `_eval_index_arg_to_shape(arg, env, warnings, ctx)`
- `expr_to_dim_ir(expr, env)` → `expr_to_dim_ir(expr, env)` (no ctx needed, uses env.dim_aliases)

### Diagnostics Changes

Add new warning functions (`analysis/diagnostics.py`):

```python
def warn_function_arg_count_mismatch(line: int, func_name: str, expected: int, got: int) -> str:
    return f"W_FUNCTION_ARG_COUNT_MISMATCH line {line}: function {func_name} expects {expected} arguments, got {got}"

def warn_recursive_function(line: int, func_name: str) -> str:
    return f"W_RECURSIVE_FUNCTION line {line}: recursive call to {func_name} not supported (returns unknown)"

def warn_multiple_returns_in_expr(line: int, func_name: str) -> str:
    return f"W_MULTIPLE_RETURNS_IN_EXPR line {line}: function {func_name} with multiple returns used in expression context"

def warn_procedure_in_expr(line: int, func_name: str) -> str:
    return f"W_PROCEDURE_IN_EXPR line {line}: procedure {func_name} has no return value, cannot be used in expression"

def warn_multi_assign_non_call(line: int) -> str:
    return f"W_MULTI_ASSIGN_NON_CALL line {line}: destructuring assignment requires function call on RHS"

def warn_multi_assign_builtin(line: int, func_name: str) -> str:
    return f"W_MULTI_ASSIGN_BUILTIN line {line}: builtin {func_name} does not support multiple returns"

def warn_multi_assign_count_mismatch(line: int, func_name: str, expected: int, got: int) -> str:
    return f"W_MULTI_ASSIGN_COUNT_MISMATCH line {line}: function {func_name} returns {expected} values, got {got} targets"
```

## Estimated Diff
- Parser: +100 lines (function parsing + destructuring parsing)
- IR: +30 lines (FunctionDef + AssignMulti nodes)
- Lowering: +50 lines (lower_function + destructuring lowering)
- Analyzer: +180 lines (AnalysisContext, function registry, analyze_function_call, AssignMulti handling, signature changes)
- Runtime: +5 lines (Env.dim_aliases field)
- Diagnostics: +25 lines (7 new warnings)
- Tests: 8 new test files (~200 lines)
- **Total: ~590 lines** across 7 files + 8 test files

## Risks

1. **Dimension aliasing breaks for non-Var args**: If caller passes `zeros(n+1, m)` as argument, `expr_to_dim_ir` extracts `(n+1)` but aliasing doesn't apply (not a Var). This is expected — aliasing only preserves simple symbolic names. Mitigated by clear documentation and test coverage.

2. **Recursion guard misses mutual recursion**: `analyzing_functions` set only guards direct recursion. Mutual recursion (A calls B, B calls A) will still loop. Mitigated by documenting limitation. Fix in v0.10.1 with call-graph depth limit.

3. **Dual-location warnings are verbose**: Function body warnings include both call site and body line. Users may find format noisy. Mitigated by clear prefix format. Can be refined based on user feedback.

4. **Parser lookahead for destructuring is fragile**: Backtracking on parse error may fail in edge cases. Mitigated by comprehensive destructuring tests. Alternative: use token-level lookahead instead of exception-based backtracking.

5. **Functions in control flow accepted silently**: MATLAB forbids `function` inside `if`/`for`/`while`. Analyzer accepts this (functions are registered in pass 1, control flow doesn't matter). Documented as design decision. May emit warning in future version if user reports confusion.

6. **AnalysisContext threading touches many signatures**: Every function that calls `eval_expr_ir` or `analyze_stmt_ir` needs `ctx` parameter. Risk of missing a call site. Mitigated by systematic grep for all call sites and test coverage.

## Deferred to v0.10.1+

- **Polymorphic functions**: Per-call-site caching based on argument shape tuple hash
- **Mutual recursion guard**: Call-graph depth limit or proper interprocedural fixpoint
- **Explicit `return` statement**: Parser support for `return` keyword (early exit from function)
- **Varargs**: `varargin`, `varargout` support
- **Function handles**: Anonymous functions `@(x) x^2` and function handles `f = @sqrt`
- **Nested functions**: Functions defined inside other functions
- **Multi-file programs**: Analyzing dependencies across .m files
- **Global variables**: `global` keyword for shared state
- **Path-sensitive function analysis**: Only re-analyze if argument shapes differ (requires shape tuple equality check)
