# W_CONTINUE_OUTSIDE_LOOP

**Severity**: Error
**Tier**: Free

Continue statement found outside a loop.

## Example

```matlab
x = 1;
continue;
```

## What this means

`continue` is only valid inside `for` or `while` loops.

## How to fix

1. Move the continue statement inside a loop.
2. Remove it if it was added by mistake.
