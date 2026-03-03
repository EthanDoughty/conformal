# W_CODER_TRY_CATCH

**Severity**: Warning
**Tier**: Coder

try/catch not supported by MATLAB Coder.

## Example

```matlab
try
  x = riskyOp();
catch e
  x = 0;
end
```

## What this means

MATLAB Coder does not support try/catch blocks.

## How to fix

1. Replace with explicit error checking or input validation.
2. Use `coder.extrinsic` to call unsupported functions outside generated code.
