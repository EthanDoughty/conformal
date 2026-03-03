# W_PROCEDURE_IN_EXPR

**Severity**: Error
**Tier**: Free

A procedure (no return value) used in an expression context.

## Example

```matlab
function greet(name)
  disp(name);
end
x = greet('hi');
```

## What this means

The function has no return values but was used where a value is expected.

## How to fix

1. Add a return value to the function if a value is needed.
2. Call it as a statement rather than in an expression.
