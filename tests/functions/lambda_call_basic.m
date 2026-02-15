% Test: Basic lambda call with shape inference
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y = matrix[3 x 1]

A = zeros(3, 1);
f = @(x) x + 1;
y = f(A);
