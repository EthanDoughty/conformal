# W_SUSPICIOUS_COMPARISON

**Severity**: Warning
**Tier**: Strict

Comparison between matrix and scalar.

## Example

```matlab
A = ones(3,4); result = A > 0;
```

## What this means

Comparing a matrix to a scalar produces a logical matrix in MATLAB, not a single boolean. This is often unintentional when used as an if-condition.

## How to fix

1. If intentional, this is valid MATLAB.
2. Use `all(A > 0, 'all')` to reduce to a scalar if a boolean result is needed.
