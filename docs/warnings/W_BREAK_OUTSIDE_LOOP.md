# W_BREAK_OUTSIDE_LOOP

**Severity**: Error
**Tier**: Free

Break statement found outside a loop.

## Example

```matlab
x = 1;
break;
```

## What this means

`break` is only valid inside `for` or `while` loops.

## How to fix

1. Move the break statement inside a loop.
2. Remove it if it was added by mistake.
