# W_POSSIBLY_NEGATIVE_DIM

**Severity**: Error
**Tier**: Pro

A dimension value may be non-positive.

## Example

```matlab
n = -1; A = zeros(n, 3);
```

## What this means

Matrix dimensions must be non-negative integers. Interval analysis determined this value could be non-positive.

## How to fix

1. Add a guard or assertion to ensure the dimension value is positive.
2. Use `max(n, 1)` to clamp the value if a minimum size of 1 is acceptable.
