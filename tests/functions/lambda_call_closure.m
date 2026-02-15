% Test: Lambda uses closure variable
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y = matrix[3 x 1]

A = zeros(3, 3);
f = @(x) A * x;
B = zeros(3, 1);
y = f(B);
