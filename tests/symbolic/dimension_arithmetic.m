% Test 6: Valid symbolic dimension multiplication, no warnings are expected
% A is m x n and B is n x m, so A * B is m x m and valid

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[4 x 3]
% EXPECT: C = matrix[3 x 3]
% EXPECT: m = scalar
% EXPECT: n = scalar

m = 3;
n = 4;
A = zeros(m, n);
B = zeros(n, m);
C = A * B;
