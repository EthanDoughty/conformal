# W_CURLY_INDEXING_NON_CELL

**Severity**: Warning
**Tier**: Default

Curly-brace indexing on a non-cell value.

## Example

```matlab
x = [1 2 3]; y = x{1};
```

## What this means

Curly-brace indexing `{...}` is only valid for cell arrays. The operand is not a cell.

## How to fix

1. Use parentheses `()` for regular matrix indexing.
2. Convert the variable to a cell array if cell indexing is intended.
