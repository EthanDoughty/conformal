% Test: Lambda closure captures environment
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: f = function_handle

A = zeros(3, 3);
f = @(x) A * x;
