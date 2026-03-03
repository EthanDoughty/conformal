# W_MULTI_ASSIGN_NON_CALL

**Severity**: Error
**Tier**: Strict

Destructuring assignment without a function call on the right side.

## Example

```matlab
[a, b] = 42;
```

## What this means

Multi-output assignment `[a, b] = expr` requires the right side to be a function call that returns multiple values.

## How to fix

1. Use a function call on the right side, e.g., `[a, b] = size(A)`.
2. Use separate assignments if the right side is a scalar or array.
