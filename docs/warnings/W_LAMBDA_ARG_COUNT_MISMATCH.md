# W_LAMBDA_ARG_COUNT_MISMATCH

**Severity**: Error
**Tier**: Free

Lambda called with wrong number of arguments.

## Example

```matlab
f = @(x, y) x + y; z = f(1);
```

## What this means

The anonymous function expects a different number of arguments than provided.

## How to fix

1. Match the number of arguments to the lambda's parameter list.
2. Review the lambda definition to ensure the parameter count is correct.
