% Test 29: size() and isscalar() builtin support
% EXPECT: warnings = 0
% EXPECT: s = matrix[1 x 2]
% EXPECT: r = scalar
% EXPECT: flag = scalar

A = zeros(3, 4);
s = size(A);
r = size(A, 1);
flag = isscalar(r);
