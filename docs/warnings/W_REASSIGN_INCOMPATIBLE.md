# W_REASSIGN_INCOMPATIBLE

**Severity**: Warning
**Tier**: Strict

Variable reassigned with incompatible shape.

## Example

```matlab
x = ones(3,4);
x = ones(5,6);
```

## What this means

A variable was reassigned with a shape that differs from its previous assignment. This may indicate a logic error.

## How to fix

1. Verify the reassignment is intentional.
2. Use a different variable name if the shapes serve different purposes.
