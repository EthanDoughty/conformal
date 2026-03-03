# W_FUNCTION_ARG_COUNT_MISMATCH

**Severity**: Error
**Tier**: Free

Function called with wrong number of arguments.

## Example

```matlab
function y = add(a, b)
  y = a + b;
end
x = add(1, 2, 3);
```

## What this means

The function was called with more or fewer arguments than its signature declares.

## How to fix

1. Match the number of arguments to the function's parameter list.
2. Use `nargin` in the function body if optional arguments are intended.
