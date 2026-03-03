# W_MATRIX_COMPARISON

**Severity**: Warning
**Tier**: Strict

Comparison between two matrices.

## Example

```matlab
A = ones(3,4); B = ones(3,4); result = A == B;
```

## What this means

Comparing two matrices produces a logical matrix, not a single boolean. This is often unintentional in conditional contexts.

## How to fix

1. Use `isequal(A, B)` for a scalar equality check.
2. If elementwise comparison is intended, the result is a logical matrix and is valid.
