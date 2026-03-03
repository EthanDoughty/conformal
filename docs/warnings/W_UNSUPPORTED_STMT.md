# W_UNSUPPORTED_STMT

**Severity**: Hint
**Tier**: Strict

Statement type not supported by the analyzer.

## Example

```matlab
parfor i = 1:10
  x(i) = i;
end
```

## What this means

Conformal does not yet analyze this statement type. Variables assigned within are still tracked but analysis is approximate.

## How to fix

1. No code change needed. This is an analysis limitation.
2. Results for variables modified inside unsupported statements should be treated as conservative.
