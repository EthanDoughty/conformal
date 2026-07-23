# W_MRDIVIDE_DIM_MISMATCH

**Severity**: Error
**Tier**: Free

Right division operands have incompatible column counts.

## Example

```matlab
A = ones(2,3); B = ones(2,4); x = A / B;
```

## What this means

For `A / B`, the number of columns in A must match the number of columns in B.

## How to fix

1. Ensure the left and right operands have the same number of columns.
2. Check whether transposing one of the operands gives the intended system.
