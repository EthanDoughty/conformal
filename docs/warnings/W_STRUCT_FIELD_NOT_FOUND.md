# W_STRUCT_FIELD_NOT_FOUND

**Severity**: Warning
**Tier**: Pro

Accessed field does not exist on the struct.

## Example

```matlab
s = struct('x', 1, 'y', 2); z = s.z;
```

## What this means

The field name used in dot-access was not found among the struct's known fields.

## How to fix

1. Check field name spelling.
2. Add the field to the struct before accessing it.
