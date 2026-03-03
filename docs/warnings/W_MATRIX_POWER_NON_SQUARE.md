# W_MATRIX_POWER_NON_SQUARE

**Severity**: Error
**Tier**: Free

Matrix power requires a square matrix.

## Example

```matlab
A = ones(3,4); B = A^2;
```

## What this means

The `^` operator for matrix exponentiation requires a square matrix. The operand here is not square.

## How to fix

1. Use a square matrix, or use `.^` for elementwise power.
2. Check whether the matrix was constructed with the intended dimensions.
