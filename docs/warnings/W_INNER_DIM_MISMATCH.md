# W_INNER_DIM_MISMATCH

**Severity**: Error
**Tier**: Free

Matrix multiplication inner dimensions do not match.

## Example

```matlab
A = ones(3,4); B = ones(5,2); C = A * B;
```

## What this means

The number of columns in the left operand must equal the number of rows in the right operand for matrix multiplication. Conformal detected that these dimensions differ.

## How to fix

1. Check matrix dimensions.
2. If you intended elementwise multiplication, use `.*` instead of `*`.
