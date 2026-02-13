# Task: Unified Apply IR Node for Runtime Call-vs-Index Disambiguation

## Goal
Replace separate Call/Index IR nodes with a unified `Apply` node. Move call-vs-index disambiguation from parse time (parser) to analysis time (analyzer), where shape information enables better decisions.

## Scope
- `/root/projects/MATLAB_analysis/ir/ir.py`: Add `Apply` IR node with `base: Expr` and `args: List[IndexArg]`
- `/root/projects/MATLAB_analysis/frontend/matlab_parser.py`: Remove builtin whitelist logic, emit unified syntax node for all `foo(...)` expressions
- `/root/projects/MATLAB_analysis/frontend/lower_ir.py`: Lower unified syntax node to `Apply` IR node
- `/root/projects/MATLAB_analysis/analysis/analysis_ir.py`: Add disambiguation logic in `eval_expr_ir` for `Apply` nodes using colon heuristic and shape information; add `unwrap_arg(IndexArg) -> Expr` helper for builtin handlers
- `/root/projects/MATLAB_analysis/analysis/diagnostics.py`: Add `Apply` node handling to `pretty_expr_ir` so warnings print useful text
- `/root/projects/MATLAB_analysis/tests/`: Add test30.m exercising runtime disambiguation (colon forces indexing, unknown base forces call warning, bound variable defaults to indexing)

## Non-goals
- Rich shape rules for phase 3 builtins (eye, randn, etc.) — deferred to Phase 3
- Path-sensitive analysis or flow-sensitive disambiguation
- Modifying legacy analyzer (remains frozen for comparison)
- Removing old Call/Index nodes immediately (keep for legacy analyzer compatibility during transition)

## Invariants Impacted
- **IR analyzer is authoritative**: Preserved — changes are localized to IR definition, lowering, and analysis
- **Parser-lowering-analyzer pipeline separation**: Strengthened — parser no longer makes semantic decisions
- **Conservative analysis soundness**: Preserved — unknown base or colon args trigger appropriate warnings/fallbacks
- **Legacy analyzer compatibility**: Preserved — old Call/Index nodes remain for legacy analyzer

## Acceptance Criteria
- [ ] New `Apply(base: Expr, args: List[IndexArg])` IR node defined in `ir/ir.py`
- [ ] Parser emits unified `['apply', line, base, args]` syntax node for all `foo(...)` expressions
- [ ] Lowering pass converts `['apply', ...]` to `Apply` IR node
- [ ] Analyzer disambiguates `Apply` nodes:
  - If any arg is `Colon` or `Range`: treat as indexing
  - If base is unknown variable: emit `W_UNKNOWN_FUNCTION`, return `unknown`
  - If base is known builtin: apply builtin shape rules
  - Otherwise: treat as indexing
- [ ] `KNOWN_BUILTINS` constant moved to analyzer (kept importable from parser for legacy compat)
- [ ] `pretty_expr_ir` in `diagnostics.py` handles `Apply` nodes
- [ ] `unwrap_arg(IndexArg) -> Expr` helper extracts inner `Expr` from `IndexExpr` args for builtin handlers
- [ ] Test30.m validates: (1) colon forces indexing, (2) unknown function emits warning, (3) builtin calls work, (4) bound variable defaults to indexing
- [ ] All existing tests pass: `python3 mmshape.py --tests`

## Commands to Run
```bash
# Run all tests
python3 /root/projects/MATLAB_analysis/mmshape.py --tests

# Run new test
python3 /root/projects/MATLAB_analysis/mmshape.py /root/projects/MATLAB_analysis/tests/test30.m

# Compare against legacy analyzer (should differ due to disambiguation change)
python3 /root/projects/MATLAB_analysis/mmshape.py --compare /root/projects/MATLAB_analysis/tests/test30.m
```

## Tests to Add/Change

**Test file**: `/root/projects/MATLAB_analysis/tests/test30.m`

Test cases:
1. **Colon forces indexing**: `B = A(:, 2);` where `A` is `matrix[3 x 4]` → result is `matrix[3 x 1]` (no warning)
2. **Builtin call works**: `C = zeros(2, 3);` → result is `matrix[2 x 3]` (no warning)
3. **Unknown function call**: `D = my_func(5);` → emit `W_UNKNOWN_FUNCTION`, result is `unknown`
4. **Range forces indexing**: `E = A(1:2, :);` → result is `matrix[2 x 4]` (no warning)
5. **Bound variable defaults to indexing**: `M = zeros(3, 4); val = M(2, 3);` → result is `scalar` (no warning, no W_UNKNOWN_FUNCTION)

Assertions:
- `% EXPECT: warnings = 1`  (only unknown function)
- `% EXPECT: A = matrix[3 x 4]`
- `% EXPECT: B = matrix[3 x 1]`
- `% EXPECT: C = matrix[2 x 3]`
- `% EXPECT: D = unknown`
- `% EXPECT: E = matrix[2 x 4]`
- `% EXPECT: M = matrix[3 x 4]`
- `% EXPECT: val = scalar`

## Migration Strategy

**Transition plan** (all old nodes retained for legacy analyzer):

1. **Phase 2a**: Add `Apply` node to IR, update parser to emit `['apply', ...]`, update lowering to handle both old and new syntax
2. **Phase 2b**: Add `Apply` handling to analyzer with disambiguation logic, keep old `Call`/`Index` handling intact
3. **Phase 2c**: Verify all tests pass with new path, validate against legacy analyzer in compare mode
4. **Phase 2d**: Document that legacy analyzer will use old Call/Index nodes, IR analyzer uses Apply

**Backward compatibility**: Legacy analyzer continues using old `Call`/`Index` nodes. Only IR analyzer uses new `Apply` node. No changes to legacy analyzer code.

## Implementation Notes

**Disambiguation algorithm** (in `eval_expr_ir` for `Apply` nodes):
```python
if isinstance(expr, Apply):
    # Check for indexing indicators
    has_colon_or_range = any(isinstance(arg, (Colon, Range)) for arg in expr.args)

    if has_colon_or_range:
        # Definitely indexing
        return handle_indexing_apply(expr, env, warnings)

    # Check if base is a known builtin function
    if isinstance(expr.base, Var):
        fname = expr.base.name
        if fname in KNOWN_BUILTINS:
            # Handle as function call
            return handle_builtin_call(fname, expr.args, expr.line, env, warnings)

        # Check if variable is unbound (unknown function)
        if fname not in env.bindings:
            warnings.append(diag.warn_unknown_function(expr.line, fname))
            return Shape.unknown()

    # Default: treat as indexing
    return handle_indexing_apply(expr, env, warnings)
```

**Parser change**: Lines 350-360 of `matlab_parser.py`
- Remove `if left[0] == "var" and left[2] in KNOWN_BUILTINS` check
- Always emit `["apply", lparen_tok.line, left, args]`

**Lowering change**: `lower_ir.py`
- Handle `"apply"` tag in `lower_expr`
- Convert to `Apply(line=expr[1], base=lower_expr(expr[2]), args=[lower_index_arg(arg) for arg in expr[3]])`
- Keep existing `"call"` and `"index"` handling for legacy compatibility

**IndexExpr unwrapping helper** (critical for builtins):
```python
def unwrap_arg(arg: IndexArg) -> Expr:
    """Extract the inner Expr from an IndexArg. Colon/End raise ValueError."""
    if isinstance(arg, IndexExpr):
        return arg.expr
    raise ValueError(f"Cannot unwrap {type(arg).__name__} to Expr")
```
All builtin handlers must use `unwrap_arg(arg)` instead of accessing `arg` directly as `Expr`.
This is the key type change: `Call.args: List[Expr]` → `Apply.args: List[IndexArg]`.

**Diagnostics change**: `diagnostics.py`
- Add `Apply` case to `pretty_expr_ir` so warnings like "indexing scalar" print `foo(...)` not `<expr>`
- For non-`Var` base, fall back to `<expr>(...)` which is sound

**Analyzer change**: `analysis_ir.py`
- Add `isinstance(expr, Apply)` case in `eval_expr_ir` with disambiguation logic above
- Define `KNOWN_BUILTINS` directly in analyzer (keep importable from parser via re-export for legacy compat)
- Extract shared indexing/call logic into helper functions
- Use `unwrap_arg()` in all builtin shape rule handlers
