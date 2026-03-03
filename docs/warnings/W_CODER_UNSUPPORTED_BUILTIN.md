# W_CODER_UNSUPPORTED_BUILTIN

**Severity**: Warning
**Tier**: Coder

Builtin function not supported by MATLAB Coder.

## Example

```matlab
s = evalc('disp(x)');
```

## What this means

The function called is not supported for code generation.

## How to fix

1. Replace with a Coder-compatible alternative.
2. See the MATLAB Coder documentation for the list of supported functions.
