# W_UNKNOWN_FUNCTION

**Severity**: Warning
**Tier**: Default

Function is not recognized by the analyzer.

## Example

```matlab
x = myCustomFunc(1, 2);
```

## What this means

The called function is not a known builtin and was not found in the workspace. Conformal cannot determine the output shape.

## How to fix

1. Ensure the function file is in the same directory as the file being analyzed.
2. Use `addpath` at the top of your script to include directories containing your functions.
3. To suppress this warning for known-good external functions, add a comment above the call:
   ```matlab
   % conformal:disable-next-line W_UNKNOWN_FUNCTION
   x = myCustomFunc(1, 2);
   ```
   Or suppress it for an entire file by adding at the top:
   ```matlab
   % conformal:disable W_UNKNOWN_FUNCTION
   ```
