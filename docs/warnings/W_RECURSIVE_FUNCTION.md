# W_RECURSIVE_FUNCTION

**Severity**: Warning
**Tier**: Pro

Recursive function call detected.

## Example

```matlab
function y = fact(n)
  if n <= 1; y = 1; else; y = n * fact(n-1); end
end
```

## What this means

Conformal detected a recursive call. Recursive analysis is not supported; the return shape is marked unknown.

## How to fix

1. No code change needed. This is an analysis limitation.
2. If shape precision is important, consider providing a manual annotation or helper.
