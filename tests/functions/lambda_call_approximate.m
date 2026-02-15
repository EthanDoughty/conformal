% Test: Calling lambda with precise body analysis
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: r = matrix[3 x 3]

f = @(x) x * x;
A = zeros(3, 3);
r = f(A);
