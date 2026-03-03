# W_RECURSIVE_LAMBDA

**Severity**: Warning
**Tier**: Strict

Recursive lambda (anonymous function) call detected.

## Example

```matlab
f = @(x) f(x-1);
```

## What this means

A lambda references itself recursively. Conformal cannot analyze recursive lambdas and marks the result as unknown.

## How to fix

1. No code change needed. This is an analysis limitation.
2. Consider converting to a named function if the recursion is intentional.
