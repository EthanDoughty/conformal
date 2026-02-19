% Test: builtins receiving Range (colon) arguments instead of plain expressions
% Regression: polyval(p, 1:10) crashed with "Cannot unwrap Range to Expr"

% EXPECT: warnings = 0

p = [1 2 3];
y = polyval(p, 1:10);
z = interp1(1:5, 1:5, 1:3);
w = sum(1:10);
