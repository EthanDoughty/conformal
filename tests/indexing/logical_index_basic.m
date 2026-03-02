A = rand(5, 3);
B = A(A > 0);  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
% EXPECT: B = matrix[None x 1]
% EXPECT: warnings = 1
