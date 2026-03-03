# W_CONCAT_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Concatenation contains a non-numeric element.

## Example

```matlab
s = struct('a', 1); A = [s, 2];
```

## What this means

Matrix concatenation expects numeric operands. A struct or other non-numeric type was found in the concatenation.

## How to fix

1. Ensure all elements in the concatenation are numeric matrices or scalars.
2. Use cell array concatenation `{s, 2}` if mixed types are intended.
