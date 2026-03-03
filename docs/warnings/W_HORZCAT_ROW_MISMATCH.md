# W_HORZCAT_ROW_MISMATCH

**Severity**: Error
**Tier**: Free

Horizontal concatenation operands have different row counts.

## Example

```matlab
A = ones(3,2); B = ones(4,2); C = [A, B];
```

## What this means

When concatenating matrices horizontally with `[A, B]`, all operands must have the same number of rows.

## How to fix

1. Ensure all matrices being concatenated horizontally have matching row counts.
2. Transpose operands if the shapes are swapped.
