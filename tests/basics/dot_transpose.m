% Test dot-transpose (non-conjugate transpose)
A = zeros(3, 4);
B = A.';
% EXPECT: B = matrix[4 x 3]

C = (A * A.').';
% EXPECT: C = matrix[3 x 3]

x = [1, 2, 3];
y = x.';
% EXPECT: y = matrix[3 x 1]
