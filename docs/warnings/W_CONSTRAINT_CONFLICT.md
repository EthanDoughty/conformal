# W_CONSTRAINT_CONFLICT

**Severity**: Error
**Tier**: Pro

Dimension constraint conflict detected.

## Example

```matlab
A = ones(3,4); B = ones(5,4); C = A * B';
```

## What this means

The constraint solver detected that a dimension variable is required to have two different concrete values simultaneously, which is impossible.

## How to fix

1. Check the shapes of all operands in the expression chain.
2. Verify that dimensions expected to match actually do.
