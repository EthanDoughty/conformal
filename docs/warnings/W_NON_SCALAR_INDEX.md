# W_NON_SCALAR_INDEX

**Severity**: Warning
**Tier**: Free

Non-scalar value used as an index argument.

## Example

```matlab
A = ones(5,5); idx = ones(2,3); x = A(idx, :);
```

## What this means

A non-scalar matrix was used as an index. Conformal conservatively marks the result as unknown shape.

## How to fix

1. No code change needed. This is an analysis limitation for complex indexing patterns.
2. If a specific output shape is required, consider using scalar indices.
