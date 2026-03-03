# W_LAMBDA_CALL_APPROXIMATE

**Severity**: Warning
**Tier**: Strict

Function handle call with limited analysis.

## Example

```matlab
f = @(x) x * 2; y = f([1 2; 3 4]);
```

## What this means

Conformal has limited ability to analyze calls through function handles. The result shape is conservatively marked unknown.

## How to fix

1. No code change needed. This is an analysis limitation for indirect function calls.
2. If shape precision is important, consider using a direct function call instead of a handle.
