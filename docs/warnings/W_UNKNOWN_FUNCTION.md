# W_UNKNOWN_FUNCTION

**Severity**: Warning
**Tier**: Pro

Function is not recognized by the analyzer.

## Example

```matlab
x = myCustomFunc(1, 2);
```

## What this means

The called function is not a known builtin and was not found in the workspace. Conformal cannot determine the output shape.

## How to fix

1. Ensure the function file is in the same directory as the file being analyzed.
2. This warning can be ignored for known-good external functions whose output shape is not critical.
