# W_CODER_CELL_ARRAY

**Severity**: Warning
**Tier**: Coder

Cell array used in code targeted for MATLAB Coder.

## Example

```matlab
c = {1, 'hello', [1 2]};
```

## What this means

MATLAB Coder has limited support for cell arrays.

## How to fix

1. Replace cell arrays with structs or regular arrays where possible.
2. If cell arrays are required, consult the MATLAB Coder documentation for supported patterns.
