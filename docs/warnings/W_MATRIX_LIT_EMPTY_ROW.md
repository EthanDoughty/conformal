# W_MATRIX_LIT_EMPTY_ROW

**Severity**: Warning
**Tier**: Free

Matrix literal contains an empty row.

## Example

```matlab
A = [1 2; ; 3 4];
```

## What this means

A row in the matrix literal has no elements. This may indicate a typo.

## How to fix

1. Remove the empty row.
2. Add the intended values for that row.
