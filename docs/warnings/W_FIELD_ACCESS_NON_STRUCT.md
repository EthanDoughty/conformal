# W_FIELD_ACCESS_NON_STRUCT

**Severity**: Warning
**Tier**: Default

Dot-access on a non-struct value.

## Example

```matlab
x = 42; y = x.field;
```

## What this means

The dot-access operator was used on a value that is not a struct.

## How to fix

1. Ensure the base variable is a struct before accessing fields.
2. Check whether the variable was assigned the wrong type earlier in the code.
