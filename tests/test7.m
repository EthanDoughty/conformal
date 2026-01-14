% Test 7: Valid scalar expansion and scalar-matrix operations, no warnings expected
% s is scalar, A is 3x4.
% A + s and s * A are both valid via scalar expansion

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: C = matrix[3 x 4]
% EXPECT: D = matrix[3 x 4]
% EXPECT: s = scalar

A = zeros(3, 4);
s = 2;
C = A + s;
D = s * A;
