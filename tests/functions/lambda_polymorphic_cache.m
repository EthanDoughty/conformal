% Test: Same lambda, different arg shapes
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y1 = matrix[3 x 1]
% EXPECT: y2 = matrix[5 x 1]

f = @(x) x + 1;
A = zeros(3, 1);
y1 = f(A);
B = zeros(5, 1);
y2 = f(B);
