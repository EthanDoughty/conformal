% Test: suppression only affects specified code, other warnings still fire
% EXPECT: warnings = 1

x = unknown_func(ones(3,3));  % conformal:disable W_UNKNOWN_FUNCTION
A = ones(3,3);
B = ones(4,4);
C = A * B;  % EXPECT_WARNING: W_INNER_DIM_MISMATCH
