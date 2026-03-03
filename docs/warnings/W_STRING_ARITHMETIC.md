# W_STRING_ARITHMETIC

**Severity**: Warning
**Tier**: Strict

Arithmetic performed on string operands.

## Example

```matlab
a = "hello"; b = a + 1;
```

## What this means

Performing arithmetic on strings may produce unexpected results. MATLAB converts strings to character codes, which is rarely intentional.

## How to fix

1. Convert strings explicitly if numeric operations are intended.
2. Use string functions like `strcat` for string manipulation instead.
