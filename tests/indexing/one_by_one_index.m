% Test: a 1x1 matrix value (e.g. a dot product) is a valid scalar index.
% It must not warn W_NON_SCALAR_INDEX, and the indexed result stays precise.
%
% EXPECT: A = matrix[5 x 5]
% EXPECT: v = matrix[5 x 1]
% EXPECT: s = matrix[1 x 1]
% EXPECT: y = scalar
% EXPECT: warnings = 0

A = zeros(5, 5);
v = A(:, 2);
s = v' * v;
y = A(s, 1);
