# W_CODER_DYNAMIC_FIELD

**Severity**: Warning
**Tier**: Coder

Dynamic struct field access not supported by MATLAB Coder.

## Example

```matlab
fname = 'x'; val = s.(fname);
```

## What this means

Dynamic field access `s.(expr)` is not supported for code generation.

## How to fix

1. Use static field names like `s.x` instead.
2. Restructure to avoid dynamic field access, for example by using a switch statement over known field names.
