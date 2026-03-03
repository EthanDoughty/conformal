# W_CELL_ASSIGN_NON_CELL

**Severity**: Warning
**Tier**: Pro

Cell assignment to a non-cell variable.

## Example

```matlab
x = 42; x{1} = 'hello';
```

## What this means

Attempting to use curly-brace assignment on a variable that is not a cell array.

## How to fix

1. Initialize the variable as a cell array first, e.g., `x = {}`.
2. Use regular indexed assignment `x(1) = value` for numeric arrays.
