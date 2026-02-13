% Test 2: Invalid matrix addition, dimensions mismatch, warning is expected
% A is 3x4 and B is 4x4, so A + B is invalid

% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[4 x 4]
% EXPECT: C = unknown

A = zeros(3, 4);
B = zeros(4, 4);
C = A + B;
