# W_CODER_RECURSION

**Severity**: Warning
**Tier**: Coder

Recursive call detected in code targeted for MATLAB Coder.

## Example

```matlab
function y = fact(n)
  if n <= 1; y = 1; else; y = n * fact(n-1); end
end
```

## What this means

MATLAB Coder supports only limited recursion with compile-time bounded depth.

## How to fix

1. Convert to an iterative implementation.
2. Use `coder.extrinsic` for non-generated code paths that require recursion.
