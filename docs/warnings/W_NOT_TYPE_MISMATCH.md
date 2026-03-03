# W_NOT_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Logical NOT applied to a non-numeric operand.

## Example

```matlab
s = struct('a',1); y = ~s;
```

## What this means

The `~` operator requires a numeric or logical operand.

## How to fix

1. Ensure the operand is numeric or logical.
2. Check whether you intended to negate a field value like `~s.a`.
