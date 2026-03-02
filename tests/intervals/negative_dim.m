% EXPECT: warnings = 1
n = -1;
A = zeros(n, 3);  % EXPECT_WARNING: W_POSSIBLY_NEGATIVE_DIM
% EXPECT: A = matrix[-1 x 3]
