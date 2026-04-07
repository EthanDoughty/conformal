# W_DIVISION_BY_ZERO

**Severity**: Error
**Tier**: Default

Division by a value that is definitely zero.

## Example

```matlab
x = 0; y = 1 / x;
```

## What this means

The divisor was proven to be zero through interval analysis. This will produce `Inf` or `NaN` at runtime.

## How to fix

1. Add a guard to check for zero before dividing.
2. Verify the divisor logic to ensure it cannot reach zero.
