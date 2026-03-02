A = ones(3, 4);
mask = A > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
% EXPECT: mask = matrix[3 x 4]
% EXPECT: warnings = 1
