# W_LOGICAL_OP_NON_SCALAR

**Severity**: Warning
**Tier**: Strict

Short-circuit logical operator used with non-scalar operands.

## Example

```matlab
A = [1 0; 1 1]; B = [1 1; 0 1]; result = A && B;
```

## What this means

The short-circuit operators `&&` and `||` expect scalar operands. Using them with matrices may cause unexpected behavior.

## How to fix

1. Use `&` or `|` for elementwise logical operations on matrices.
2. Use `all(A, 'all') && all(B, 'all')` if a scalar boolean result is needed.
