# W_EXTERNAL_PARSE_ERROR

**Severity**: Warning
**Tier**: Default

Parse error in an external function file.

## Example

```matlab
% In main.m:
y = helper(x);
% helper.m has a syntax error
```

## What this means

Conformal attempted to analyze a cross-file function call but encountered a parse error in the external file.

## How to fix

1. Fix the syntax error in the referenced file.
2. Run Conformal on that file directly to see the specific parse error.
