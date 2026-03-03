# W_ARITHMETIC_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Arithmetic operator applied to non-numeric operands.

## Example

```matlab
s = struct('a',1); x = s + 1;
```

## What this means

Arithmetic operators (`+`, `-`, `*`, etc.) require numeric operands. A non-numeric type (struct, cell, string) was detected.

## How to fix

1. Convert operands to numeric types before performing arithmetic.
2. Check whether you intended to access a field of the struct instead.
