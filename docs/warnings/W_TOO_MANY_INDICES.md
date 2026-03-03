# W_TOO_MANY_INDICES

**Severity**: Warning
**Tier**: Strict

More than two indices used on a 2D matrix.

## Example

```matlab
A = ones(3,4); x = A(1,2,3);
```

## What this means

Conformal's shape domain is 2D. Indexing with more than two subscripts exceeds its tracking capability. The result shape is conservatively marked unknown.

## How to fix

1. This is a Conformal analysis limitation. If your code uses 3D+ arrays, this warning can be safely ignored.
2. Consider restructuring to use 2D indexing if the extra dimension is not needed.
