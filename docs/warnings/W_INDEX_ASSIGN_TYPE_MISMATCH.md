# W_INDEX_ASSIGN_TYPE_MISMATCH

**Severity**: Error
**Tier**: Free

Indexed assignment to a non-indexable value.

## Example

```matlab
x = 'hello'; x(1) = struct('a',1);
```

## What this means

Attempting to perform indexed assignment on a value whose type does not support indexing.

## How to fix

1. Ensure the target variable has an indexable type (matrix, cell, etc.).
2. Reinitialize the variable with the correct type before performing indexed assignment.
