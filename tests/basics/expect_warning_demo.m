% Test: inline EXPECT_WARNING directive
% EXPECT: warnings = 1
% EXPECT: y = unknown
A = zeros(3, 4);
x = zeros(5, 1);
y = A * x;  % EXPECT_WARNING: W_INNER_DIM_MISMATCH
