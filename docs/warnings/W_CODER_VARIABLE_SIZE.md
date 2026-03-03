# W_CODER_VARIABLE_SIZE

**Severity**: Warning
**Tier**: Coder

Variable has unbounded dimensions for MATLAB Coder.

## Example

```matlab
x = zeros(n, 1);  % n is not compile-time constant
```

## What this means

MATLAB Coder requires fixed-size or explicitly declared variable-size arrays.

## How to fix

1. Use `coder.varsize('x')` to declare the variable as variable-size.
2. Ensure dimensions are compile-time constants where possible.
