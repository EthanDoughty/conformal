% Concrete dimension mismatch — trivial witness (no symbolic vars needed).
A = zeros(3, 4);
B = zeros(5, 2);
C = A * B;  % EXPECT_WARNING: W_INNER_DIM_MISMATCH
% EXPECT: warnings = 1
