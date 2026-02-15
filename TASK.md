# Task: Language Extensions v0.12.0 — Strings, Structs, Anonymous Functions

## Goal
Expand the Mini-MATLAB subset with three new language features: string/char-array literals, struct support with field access, and anonymous functions/function handles. Each feature requires parser changes, new IR nodes, lowering logic, shape domain extensions, analyzer semantics, and comprehensive tests.

Phase 4 (nested functions) is deferred to v0.13.0 due to complexity.

## Phasing Strategy

**Phase 1: Strings** (Simplest)
- Context-sensitive lexing: `'` is string start after operators/`=`/`(`/`;`/`,`/`[`; transpose after ID/`)` /`]`/NUMBER
- Both `'hello'` and `"hello"` supported via context-sensitive lexer
- MATLAB-faithful `+` on strings: produces numeric row vector (not string concat)
- String concatenation via horzcat: `['hello', ' ', 'world']`
- New shape kind: `Shape.string()`

**Phase 2: Structs** (Moderate)
- New shape kind: `Shape.struct(fields)` with sorted-tuple storage for hashability
- Chained dot access from the start: `s.a.b = 5`, `x = s.a.b`
- Union-with-bottom join semantics (missing fields get `Shape.bottom()`, not `unknown`)
- Promotes `tests/recovery/struct_field.m` from `W_UNSUPPORTED_STMT` to `W_FIELD_ACCESS_NON_STRUCT`

**Phase 3: Anonymous Functions** (Moderate-High)
- New shape kind: `Shape.function_handle()`
- `@(x) expr` lambda syntax and `@myFunc` named function handles
- Function handle variables shadow builtins in Apply disambiguation
- Lambda calls return `unknown` with `W_LAMBDA_CALL_APPROXIMATE` (body analysis deferred to v0.12.1)

**Order**: Phase 1 → Phase 2 → Phase 3. Each phase passes all tests before starting the next.

## Design Decisions (from review)

1. **String token**: Context-sensitive lexing. Track previous token to disambiguate `'` as string start vs transpose.
2. **Struct join**: Union with bottom. Missing fields get `Shape.bottom()` (identity in join). Precise and sound.
3. **Struct depth**: Chained from start. `StructAssign(base_name: str, fields: List[str], expr: Expr)`.
4. **String `+`**: Follows MATLAB. `'hello' + ' '` produces numeric vector. String concat via `['a', 'b']` horzcat.
5. **Handle shadows builtins**: Yes. Function handle variable overrides `KNOWN_BUILTINS` lookup. Matches real MATLAB scoping.
6. **Struct hashability**: Sorted tuple of pairs: `fields: Tuple[Tuple[str, Shape], ...]` with a `fields_dict` property.
7. **Lambda closure key**: Monotonic counter in AnalysisContext (not line number, avoids collision).
8. **`struct_field.m` expectation**: Keep `warnings = 1`, keep `B = unknown`. Warning code changes from `W_UNSUPPORTED_STMT` to `W_FIELD_ACCESS_NON_STRUCT`.

## Scope

### Phase 1: Strings

**Lexer changes** (`matlab_parser.py`):
- Add `DQSTRING` token: `r'"[^"]*"'` — always a double-quoted string (no ambiguity)
- For single-quoted strings: context-sensitive `'` disambiguation in `lex()`:
  - After ID, `)`, `]`, NUMBER, TRANSPOSE → emit `TRANSPOSE` token
  - After OP, `=`, `(`, `,`, `;`, `[`, NEWLINE, start-of-file, keywords → scan ahead for matching `'`, emit `STRING` token with contents
  - Both `STRING` and `DQSTRING` tokens map to the same `StringLit` IR node
- Remove the standalone `TRANSPOSE` regex entry (handled in context-sensitive logic)

**IR changes** (`ir/ir.py`):
```python
@dataclass(frozen=True)
class StringLit(Expr):
    """String literal ('hello' or "world")."""
    value: str
```

**Lowering** (`lower_ir.py`):
- Add case for `"string"` tag → `StringLit(line=..., value=...)`

**Shape domain** (`shapes.py`):
- Add `Shape.string()` constructor: `Shape(kind="string")`
- Add `is_string()` predicate
- Update `join_shape`: string+string=string, string+other=unknown
- Update `widen_shape`: same as join
- Update `__str__`: return `"string"` for string kind

**Analyzer** (`analysis_ir.py`):
- `StringLit` handler in `eval_expr_ir` → return `Shape.string()`
- BinOp `+` with two strings → `Shape.matrix(1, None)` (MATLAB: char + char = numeric row vector, length unknown without concrete tracking)
- BinOp with string + matrix/scalar → emit `W_STRING_ARITHMETIC` warning, return `Shape.unknown()`
- MatrixLit with string elements → string in row = `Shape.string()` as scalar-like element (horzcat of strings is valid MATLAB)

**Diagnostics** (`diagnostics.py`):
- `warn_string_arithmetic(line, op, left_shape, right_shape)` → `W_STRING_ARITHMETIC`

**Tests** (4 files in `tests/literals/`):
1. `string_literal.m`: Basic string assignment → `Shape.string()`
2. `string_horzcat.m`: `s = ['hello', ' ', 'world']` → tests horzcat with strings
3. `string_matrix_error.m`: `A + 'error'` → warning + unknown
4. `string_in_control_flow.m`: String/scalar join → unknown

### Phase 2: Structs

**Parser changes** (`matlab_parser.py`):
- `parse_postfix` case for dot access: after `DOT`, eat `ID` → `["field_access", line, base, field_name]`
- Distinguish `A.field` from `A.*B`: DOT followed by ID → field access (existing `DOTOP` token already handles `.*` and `./`)
- Struct field assignment in `parse_simple_stmt`: detect `ID DOT ID (DOT ID)* = expr` pattern → `["struct_assign", line, base_name, [field1, field2, ...], rhs_expr]`

**IR changes** (`ir/ir.py`):
```python
@dataclass(frozen=True)
class FieldAccess(Expr):
    """Struct field access (s.field or s.a.b)."""
    base: Expr
    field: str  # Outermost field in chain parsed as nested FieldAccess nodes

@dataclass(frozen=True)
class StructAssign(Stmt):
    """Struct field assignment (s.field = expr or s.a.b = expr)."""
    base_name: str
    fields: List[str]  # Chain of field names [a, b] for s.a.b = expr
    expr: Expr
```

Note: `FieldAccess` is nested for chained reads (`s.a.b` → `FieldAccess(FieldAccess(Var("s"), "a"), "b")`).
`StructAssign` uses a flat field list since the LHS is always `ID.field.field...`.

**Lowering** (`lower_ir.py`):
- `"field_access"` tag → `FieldAccess(line, base=lower_expr(base), field=field_name)`
- `"struct_assign"` tag → `StructAssign(line, base_name, fields, lower_expr(rhs))`

**Shape domain** (`shapes.py`):
```python
@staticmethod
def struct(fields: dict) -> "Shape":
    """Create a struct shape with given fields.
    Stored as sorted tuple of pairs for hashability.
    """
    return Shape(kind="struct", rows=None, cols=None,
                 _fields=tuple(sorted(fields.items())))

@property
def fields_dict(self) -> dict:
    """Get fields as a dict (for struct shapes only)."""
    return dict(self._fields) if self.kind == "struct" else {}
```

- Add `_fields: Tuple[Tuple[str, "Shape"], ...] = ()` to Shape dataclass
- Add `is_struct()` predicate
- Update `join_shape` for structs:
  - Same field set → pointwise `join_shape` on field values
  - Different field sets → union of all field names, missing fields use `Shape.bottom()` (identity in join), then pointwise join
- Update `widen_shape` similarly
- Update `__str__`: `"struct{x: scalar, y: matrix[3 x 1]}"` format

**Analyzer** (`analysis_ir.py`):
- `FieldAccess` handler in `eval_expr_ir`:
  - Evaluate base → get shape
  - If struct: look up field in `fields_dict`, return field shape (or warn `W_STRUCT_FIELD_NOT_FOUND` if missing)
  - If not struct: warn `W_FIELD_ACCESS_NON_STRUCT`, return `unknown`
- `StructAssign` handler in `analyze_stmt_ir`:
  - Evaluate RHS expr
  - Get current base variable shape (may be `bottom` if unbound)
  - Walk the field chain: for `s.a.b = expr`, update nested struct shape
  - Set base variable to updated struct
  - If base is bottom → create fresh struct from chain
  - If base is struct → update existing struct (preserving other fields)
  - If base is non-struct → warn, treat as fresh struct

**Diagnostics** (`diagnostics.py`):
- `warn_struct_field_not_found(line, field, struct_shape)` → `W_STRUCT_FIELD_NOT_FOUND`
- `warn_field_access_non_struct(line, base_shape)` → `W_FIELD_ACCESS_NON_STRUCT`

**Tests** (5 files in `tests/structs/` — new directory):
1. `struct_create_assign.m`: Create struct via field assignment → `struct{x: scalar, y: matrix[3 x 1]}`
2. `struct_field_access.m`: Read struct fields → extracts correct shapes
3. `struct_field_not_found.m`: Access non-existent field → warning + unknown
4. `struct_in_control_flow.m`: Struct joined in if/else branches → pointwise field join
5. `struct_field_reassign.m`: Reassign field with different shape → `struct{x: matrix[3 x 3]}` (was scalar)

**Existing test update**:
- `tests/recovery/struct_field.m`: Keep `warnings = 1`, keep `B = unknown`. Warning code changes from `W_UNSUPPORTED_STMT` to `W_FIELD_ACCESS_NON_STRUCT`. The test now exercises the first-class field access path instead of recovery.

### Phase 3: Anonymous Functions

**Lexer changes** (`matlab_parser.py`):
- Add `@` to the OP token pattern (add `@` to the character class in OP regex)

**Parser changes** (`matlab_parser.py`):
- In `parse_expr` prefix handling, when `@` is encountered:
  - If next token is `(` → anonymous function: `@(params) body_expr` → `["lambda", line, params, body_expr]`
  - If next token is `ID` → named function handle: `@myFunc` → `["func_handle", line, name]`
- New `parse_anonymous_function` method:
  ```python
  def parse_anonymous_function(self):
      at_tok = self.eat("@")
      self.eat("(")
      params = []
      if self.current().value != ")":
          params.append(self.eat("ID").value)
          while self.current().value == ",":
              self.eat(",")
              params.append(self.eat("ID").value)
      self.eat(")")
      body = self.parse_expr()
      return ["lambda", at_tok.line, params, body]
  ```

**IR changes** (`ir/ir.py`):
```python
@dataclass(frozen=True)
class Lambda(Expr):
    """Anonymous function (@(x) x+1 or @(x,y) x+y)."""
    params: List[str]
    body: Expr

@dataclass(frozen=True)
class FuncHandle(Expr):
    """Named function handle (@myFunc)."""
    name: str
```

**Lowering** (`lower_ir.py`):
- `"lambda"` tag → `Lambda(line, params, lower_expr(body))`
- `"func_handle"` tag → `FuncHandle(line, name)`

**Shape domain** (`shapes.py`):
- Add `Shape.function_handle()` constructor: `Shape(kind="function_handle")`
- Add `is_function_handle()` predicate
- Update `join_shape`: fh+fh=fh, fh+other=unknown
- Update `widen_shape`: same
- Update `__str__`: return `"function_handle"` for fh kind

**Analyzer** (`analysis_ir.py`):
- `Lambda` handler in `eval_expr_ir`:
  - Snapshot current env as closure (store in `ctx._lambda_closures[ctx._next_lambda_id]`, increment counter)
  - Return `Shape.function_handle()`
- `FuncHandle` handler in `eval_expr_ir`:
  - Check if name is in `ctx.function_registry` or `KNOWN_BUILTINS` → return `Shape.function_handle()`
  - Otherwise → warn `W_UNKNOWN_FUNCTION`, return `Shape.function_handle()`
- Update Apply disambiguation order:
  1. Colon/Range in args → force indexing
  2. **Check base variable shape is `function_handle` → function call (return `unknown`, emit `W_LAMBDA_CALL_APPROXIMATE`)**
  3. Check base is known builtin → builtin call
  4. Check base in function_registry → user-defined function call
  5. Otherwise → indexing or unknown function
- Add to AnalysisContext:
  ```python
  _lambda_closures: Dict[int, Env] = field(default_factory=dict)
  _next_lambda_id: int = 0
  ```

**Diagnostics** (`diagnostics.py`):
- `warn_lambda_call_approximate(line, var_name)` → `W_LAMBDA_CALL_APPROXIMATE`

**Tests** (5 files in `tests/functions/`):
1. `lambda_basic.m`: Anonymous function assignment → `Shape.function_handle()`
2. `lambda_call_approximate.m`: Calling lambda → warning + unknown
3. `lambda_zero_args.m`: `f = @() 42` → function_handle (zero-arg lambda)
4. `function_handle_from_name.m`: `f = @myFunc` → function_handle
5. `function_handle_join.m`: `if c; f = @(x) x; else; f = 5; end` → unknown (fh join scalar)

## Non-goals

- **Phase 4 (nested functions)**: Deferred to v0.13.0
- **Lambda body analysis**: v0.12.0 treats all lambda calls as `unknown`; interprocedural lambda inference deferred to v0.12.1
- **Struct literal syntax**: `struct('x', 1, 'y', 2)` constructor deferred
- **Cell arrays**: Still in recovery mode
- **String indexing/comparison**: No `s(1:3)`, no `strcmp`
- **Function handle signatures**: No arg count validation when calling function handles
- **Dynamic struct arrays**: No `s(i).field`

## Invariants Impacted

- **Shape domain**: Extended from 4 kinds (scalar, matrix, unknown, bottom) to 7 (+ string, struct, function_handle)
  - All new kinds implement join/widen semantics (Preserved)
  - `Shape` remains frozen dataclass (Preserved — struct uses sorted tuple)
- **IR node coverage**: New expr nodes (StringLit, FieldAccess, Lambda, FuncHandle) and stmt nodes (StructAssign) all have lowering and analysis handlers (Preserved)
- **Recovery mode**: Struct field access promoted from OpaqueStmt → first-class (Tightened)
- **Test discovery**: New `tests/structs/` directory (9 → 10 categories) (Extended)
- **Analyzer soundness**: Conservative unknown fallback for unsupported cases (Preserved)
- **Apply disambiguation**: Function handle check inserted before builtin check (Changed — shadows builtins)

## Acceptance Criteria

### Phase 1: Strings
- [ ] Context-sensitive lexer disambiguates `'` as string vs transpose
- [ ] Both `'hello'` and `"hello"` parse as StringLit
- [ ] `Shape.string()` with join/widen support
- [ ] `+` on two strings → `Shape.matrix(1, None)` (numeric vector)
- [ ] String + matrix → warning + unknown
- [ ] All 4 string tests pass
- [ ] All 92 existing tests pass

### Phase 2: Structs
- [ ] Dot field access parsed correctly (doesn't break `.*` operator)
- [ ] Chained access works: `s.a.b` reads and `s.a.b = expr` assigns
- [ ] `Shape.struct(fields)` with sorted-tuple storage is hashable
- [ ] Union-with-bottom join: missing fields get `Shape.bottom()`
- [ ] `W_STRUCT_FIELD_NOT_FOUND` and `W_FIELD_ACCESS_NON_STRUCT` warnings
- [ ] `tests/recovery/struct_field.m` now uses `W_FIELD_ACCESS_NON_STRUCT` (not `W_UNSUPPORTED_STMT`)
- [ ] All 5 struct tests pass
- [ ] All previous tests pass

### Phase 3: Anonymous Functions
- [ ] `@(x) expr` and `@myFunc` parse correctly
- [ ] `Shape.function_handle()` with join/widen support
- [ ] Function handle variables shadow builtins in Apply disambiguation
- [ ] Lambda calls emit `W_LAMBDA_CALL_APPROXIMATE` and return unknown
- [ ] All 5 lambda tests pass
- [ ] All previous tests pass

## Commands to Run

```bash
# Phase 1
python3 mmshape.py tests/literals/string_literal.m
python3 mmshape.py tests/literals/string_horzcat.m
python3 mmshape.py tests/literals/string_matrix_error.m
python3 mmshape.py tests/literals/string_in_control_flow.m
python3 mmshape.py --tests

# Phase 2
python3 mmshape.py tests/structs/struct_create_assign.m
python3 mmshape.py tests/structs/struct_field_access.m
python3 mmshape.py tests/structs/struct_field_not_found.m
python3 mmshape.py tests/structs/struct_in_control_flow.m
python3 mmshape.py tests/structs/struct_field_reassign.m
python3 mmshape.py tests/recovery/struct_field.m
python3 mmshape.py --tests

# Phase 3
python3 mmshape.py tests/functions/lambda_basic.m
python3 mmshape.py tests/functions/lambda_call_approximate.m
python3 mmshape.py tests/functions/lambda_zero_args.m
python3 mmshape.py tests/functions/function_handle_from_name.m
python3 mmshape.py tests/functions/function_handle_join.m
python3 mmshape.py --tests

# All modes
python3 mmshape.py --fixpoint --tests
```

## Tests to Add

**Phase 1 (Strings)**: 4 tests in `tests/literals/`

1. `tests/literals/string_literal.m`:
```matlab
% Test: String literals (both quote styles)
% EXPECT: warnings = 0
% EXPECT: s = string
% EXPECT: t = string

s = 'hello';
t = "world";
```

2. `tests/literals/string_horzcat.m`:
```matlab
% Test: String concatenation via horzcat
% EXPECT: warnings = 0
% EXPECT: s = string

s = ['hello', ' ', 'world'];
```

3. `tests/literals/string_matrix_error.m`:
```matlab
% Test: String-matrix arithmetic produces warning
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]
% EXPECT: r = unknown

A = zeros(3, 3);
r = A + 'error';
```

4. `tests/literals/string_in_control_flow.m`:
```matlab
% Test: String/scalar join across branches
% EXPECT: warnings = 0
% EXPECT: v = unknown

if 1
    v = 'hello';
else
    v = 5;
end
```

**Phase 2 (Structs)**: 5 tests in `tests/structs/` (new directory)

1. `tests/structs/struct_create_assign.m`:
```matlab
% Test: Create struct via field assignment
% EXPECT: warnings = 0
% EXPECT: s = struct{x: scalar, y: matrix[3 x 1]}

s.x = 5;
s.y = zeros(3, 1);
```

2. `tests/structs/struct_field_access.m`:
```matlab
% Test: Read struct fields
% EXPECT: warnings = 0
% EXPECT: s = struct{a: matrix[2 x 2], b: scalar}
% EXPECT: A = matrix[2 x 2]
% EXPECT: val = scalar

s.a = zeros(2, 2);
s.b = 10;
A = s.a;
val = s.b;
```

3. `tests/structs/struct_field_not_found.m`:
```matlab
% Test: Access non-existent field
% EXPECT: warnings = 1
% EXPECT: s = struct{x: scalar}
% EXPECT: r = unknown

s.x = 5;
r = s.y;
```

4. `tests/structs/struct_in_control_flow.m`:
```matlab
% Test: Struct shapes joined in branches (union with bottom)
% If-branch has {x, y}, else has {x} → join gives {x: scalar, y: bottom→scalar}
% But bottom is filtered from display, so y appears as scalar from if-branch
% EXPECT: warnings = 0
% EXPECT: s = struct{x: scalar, y: scalar}

s.x = 1;
if 1
    s.x = 5;
    s.y = 10;
else
    s.x = 3;
end
```

5. `tests/structs/struct_field_reassign.m`:
```matlab
% Test: Reassign field with different shape
% EXPECT: warnings = 0
% EXPECT: s = struct{x: matrix[3 x 3]}

s.x = 5;
s.x = zeros(3, 3);
```

**Phase 3 (Anonymous Functions)**: 5 tests in `tests/functions/`

1. `tests/functions/lambda_basic.m`:
```matlab
% Test: Anonymous function assignment
% EXPECT: warnings = 0
% EXPECT: f = function_handle

f = @(x) x + 1;
```

2. `tests/functions/lambda_call_approximate.m`:
```matlab
% Test: Calling lambda returns unknown (v0.12.0 limitation)
% EXPECT: warnings = 1
% EXPECT: f = function_handle
% EXPECT: r = unknown

f = @(x) x * x;
A = zeros(3, 3);
r = f(A);
```

3. `tests/functions/lambda_zero_args.m`:
```matlab
% Test: Zero-argument lambda
% EXPECT: warnings = 0
% EXPECT: f = function_handle

f = @() 42;
```

4. `tests/functions/function_handle_from_name.m`:
```matlab
% Test: Function handle from named function
% EXPECT: warnings = 0
% EXPECT: f = function_handle

function y = myFunc(x)
    y = x + 1;
end

f = @myFunc;
```

5. `tests/functions/function_handle_join.m`:
```matlab
% Test: Function handle joined with scalar → unknown
% EXPECT: warnings = 0
% EXPECT: f = unknown

if 1
    f = @(x) x;
else
    f = 5;
end
```

## Estimated Diff

**Phase 1 (Strings)**: ~120 lines
- Lexer: +30 lines (context-sensitive `'`, `DQSTRING` token)
- IR: +6 lines (StringLit)
- Lowering: +4 lines
- Shapes: +25 lines (string kind, join/widen, __str__)
- Analyzer: +30 lines (StringLit eval, string arithmetic)
- Diagnostics: +5 lines
- Tests: +30 lines (4 files)

**Phase 2 (Structs)**: ~350 lines
- Parser: +40 lines (dot access postfix, struct assign detection)
- IR: +16 lines (FieldAccess, StructAssign)
- Lowering: +20 lines
- Shapes: +80 lines (struct kind, _fields, join/widen with union-bottom, __str__)
- Analyzer: +100 lines (FieldAccess eval, StructAssign handler, nested struct update)
- Diagnostics: +15 lines
- Tests: +70 lines (5 files + 1 existing update)

**Phase 3 (Anonymous Functions)**: ~200 lines
- Lexer/Parser: +30 lines (@ token, parse_anonymous_function, parse_func_handle)
- IR: +12 lines (Lambda, FuncHandle)
- Lowering: +8 lines
- Shapes: +20 lines (function_handle kind, join/widen, __str__)
- Analyzer: +60 lines (Lambda eval, FuncHandle eval, Apply disambiguation update, closure tracking)
- Diagnostics: +5 lines
- Tests: +60 lines (5 files)

**Grand Total (Phases 1-3)**: ~670 lines, 14 new tests, 1 existing test updated
**Test count**: 92 → 106 (+14)
**Test categories**: 9 → 10 (new `structs/`)

## Risks

**R1 (HIGH). Single-quote context-sensitive lexing.** The lexer must track previous token type to decide if `'` starts a string or is transpose. This is the most complex lexer change to date. Edge cases: `A'*B` (transpose then multiply), `f('hello')` (string arg to function), `A(:)'` (transpose of slice). Mitigation: thorough testing, keep the context table minimal and well-documented.

**R2 (MEDIUM). Shape kind combinatorial explosion.** 7 shape kinds means 49 pairwise combinations in join/widen. Use a "same-kind" fast path and default to `unknown` for all cross-kind joins. Document the join table explicitly.

**R3 (MEDIUM). Struct assign parser detection.** Must detect `ID.field.field... = expr` in `parse_simple_stmt` without breaking existing `ID = expr` or `ID(args) = expr` patterns. The parser eats ID first, then checks for DOT.

**R4 (LOW). Lambda closure correctness.** v0.12.0 stores closures but doesn't use them (lambda calls return unknown). Closure correctness is deferred to v0.12.1 body analysis.

**R5 (LOW). Frozen dataclass with new fields.** Adding `_fields` to Shape requires updating the dataclass. Since Shape is frozen, all construction sites must be reviewed. The default `_fields=()` means existing code works unchanged.

## Version Numbering

- **v0.12.0**: Phases 1-3 (strings, structs, anonymous functions)
- **v0.12.1** (future): Lambda body analysis, single-quote edge case hardening
- **v0.13.0** (future): Nested functions with shared workspace
