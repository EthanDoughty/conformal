# W_NEGATE_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Negation applied to a non-numeric operand.

## Example

```matlab
s = struct('a',1); y = -s;
```

## What this means

The unary minus operator requires a numeric operand.

## How to fix

1. Ensure the operand is numeric before negating.
2. Check whether you intended to negate a field value like `-s.a`.
