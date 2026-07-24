# W_NON_SCALAR_INDEX

**Severity**: Warning
**Tier**: Free

A subscript that is provably neither scalar nor a vector was used as an index argument.

## Example

```matlab
A = ones(5,5); idx = ones(2,3); x = A(idx, :);
```

## What this means

A matrix subscript (both dimensions concrete and greater than 1) was used as an index. Conformal cannot determine how MATLAB linearizes this subscript, so the extent along that dimension is assumed unknown; the other dimension is still tracked.

## How to fix

1. No code change needed. This is an analysis limitation for complex indexing patterns.
2. If a specific output shape is required, consider using scalar or vector indices.
