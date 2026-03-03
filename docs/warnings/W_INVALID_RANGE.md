# W_INVALID_RANGE

**Severity**: Warning
**Tier**: Free

Range end is less than start.

## Example

```matlab
x = 5:1;
```

## What this means

A range with end less than start produces an empty array in MATLAB.

## How to fix

1. Swap the endpoints if a decreasing sequence was intended.
2. Use a negative step like `5:-1:1` for a decreasing range.
