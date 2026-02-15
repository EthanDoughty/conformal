% Test: Calling lambda returns unknown (v0.12.0 limitation)
% EXPECT: warnings = 1
% EXPECT: f = function_handle
% EXPECT: r = unknown

f = @(x) x * x;
A = zeros(3, 3);
r = f(A);
