# W_CELLFUN_NON_UNIFORM

**Severity**: Warning
**Tier**: Strict

cellfun produces non-scalar output without UniformOutput=false.

## Example

```matlab
c = {[1 2], [3 4]}; r = cellfun(@(x) x*2, c);
```

## What this means

`cellfun` expects scalar outputs by default. Non-scalar results require `'UniformOutput', false` to collect into a cell array.

## How to fix

1. Add `'UniformOutput', false` to the cellfun call.
2. If scalar output is expected, verify the function handle returns a scalar.
