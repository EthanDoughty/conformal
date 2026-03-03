# W_MULTI_ASSIGN_COUNT_MISMATCH

**Severity**: Error
**Tier**: Free

Number of assignment targets does not match function return count.

## Example

```matlab
function y = single_out(x)
  y = x;
end
[a, b] = single_out(1);
```

## What this means

The function returns fewer values than the number of variables on the left side of the assignment.

## How to fix

1. Match the number of output variables to the function's return count.
2. Use `~` to discard unwanted outputs if needed: `[a, ~] = twoOutputFunc()`.
