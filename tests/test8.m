% Test 8: Invalid elementwise operation mismatch, warning is expected
% A is 3x4 and B is 3x5, so A .* B is invalid due to differing column sizes

% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 5]
% EXPECT: C = matrix[3 x None]

A = zeros(3, 4);
B = zeros(3, 5);
C = A .* B;
