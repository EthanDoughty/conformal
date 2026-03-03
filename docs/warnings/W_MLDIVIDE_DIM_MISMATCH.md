# W_MLDIVIDE_DIM_MISMATCH

**Severity**: Error
**Tier**: Free

Left division operands have incompatible row counts.

## Example

```matlab
A = ones(4,3); b = ones(5,1); x = A \ b;
```

## What this means

For `A \ b`, the number of rows in A must match the number of rows in b.

## How to fix

1. Ensure the left and right operands have the same number of rows.
2. Check whether transposing one of the operands gives the intended system.
