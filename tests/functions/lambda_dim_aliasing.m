% Test: Dimension aliasing through lambda calls
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: B = matrix[5 x 5]

f = @(n) zeros(n, n);
B = f(5);
