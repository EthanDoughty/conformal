# W_VERTCAT_COL_MISMATCH

**Severity**: Error
**Tier**: Free

Vertical concatenation operands have different column counts.

## Example

```matlab
A = ones(3,2); B = ones(3,4); C = [A; B];
```

## What this means

When concatenating matrices vertically with `[A; B]`, all operands must have the same number of columns.

## How to fix

1. Ensure all matrices being concatenated vertically have matching column counts.
2. Transpose operands if the shapes are swapped.
