# W_RETURN_OUTSIDE_FUNCTION

**Severity**: Warning
**Tier**: Free

Return statement found outside a function body.

## Example

```matlab
x = 1;
return;
```

## What this means

A `return` statement was found in a script context rather than inside a function.

## How to fix

1. Move the return statement inside a function.
2. Remove it from the script if it was added by mistake.
