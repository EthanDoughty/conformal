# W_ELEMENTWISE_MISMATCH

**Severity**: Error
**Tier**: Free

Elementwise operation operands have incompatible shapes.

## Example

```matlab
A = ones(3,4); B = ones(3,5); C = A + B;
```

## What this means

Elementwise operations require both operands to have the same shape, or one must be scalar. The shapes here differ in at least one dimension.

## How to fix

1. Verify both operands have the same dimensions.
2. Use scalar expansion where appropriate.
