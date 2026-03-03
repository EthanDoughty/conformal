# W_INDEX_OUT_OF_BOUNDS

**Severity**: Error
**Tier**: Pro

Index exceeds the dimension size.

## Example

```matlab
A = ones(3,4); x = A(5, 1);
```

## What this means

The index value exceeds the known dimension size. This will cause a runtime error in MATLAB.

## How to fix

1. Check that all indices are within the valid range for each dimension.
2. Add bounds checks or use `min(idx, size(A, dim))` to clamp indices.
