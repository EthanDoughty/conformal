# W_RANGE_NON_SCALAR

**Severity**: Warning
**Tier**: Free

Range endpoints in indexing are not scalar.

## Example

```matlab
A = ones(5,5); r = [1 2]; x = A(r:4, :);
```

## What this means

Range expressions like `a:b` require scalar endpoints. A non-scalar value was used as a range bound.

## How to fix

1. Ensure range endpoints are scalar values.
2. Use explicit scalar indexing or `end` for boundary expressions.
