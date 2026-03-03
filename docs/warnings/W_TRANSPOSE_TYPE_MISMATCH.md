# W_TRANSPOSE_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Transpose applied to a non-numeric operand.

## Example

```matlab
s = struct('a',1); y = s';
```

## What this means

The transpose operator requires a numeric matrix or scalar operand.

## How to fix

1. Ensure the operand is numeric before transposing.
2. Check whether you intended to access a field of the struct instead.
