% Test 1: Valid matrix addition, no warnings are expected
% A and B are both 3x4, so C = A + B is valid

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 4]
% EXPECT: C = matrix[3 x 4]

A = zeros(3, 4);
B = ones(3, 4);
C = A + B;
