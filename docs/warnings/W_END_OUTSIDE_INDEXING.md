# W_END_OUTSIDE_INDEXING

**Severity**: Warning
**Tier**: Free

`end` keyword used outside an indexing expression.

## Example

```matlab
x = end;
```

## What this means

The `end` keyword is only valid inside indexing expressions like `A(1:end)`.

## How to fix

1. Use `end` only within indexing subscripts.
2. Replace with an explicit size expression like `size(A, 1)` if a dimension value is needed.
