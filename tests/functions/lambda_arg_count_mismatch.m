% Test: Lambda called with wrong arg count
% EXPECT: warnings = 1
% EXPECT: f = function_handle
% EXPECT: z = unknown

f = @(x, y) x + y;
A = zeros(3, 1);
z = f(A);
