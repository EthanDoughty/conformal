# W_RESHAPE_MISMATCH

**Severity**: Error
**Tier**: Free

Reshape target size does not match element count.

## Example

```matlab
A = ones(3,4); B = reshape(A, 2, 5);
```

## What this means

The total number of elements in the reshaped matrix must equal the original. A 3x4 matrix has 12 elements, but 2x5 = 10.

## How to fix

1. Ensure the product of target dimensions equals the number of elements in the source.
2. Use `numel(A)` to verify element counts before reshaping.
