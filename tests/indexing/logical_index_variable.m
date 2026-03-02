A = rand(3, 3);
mask = A > 0.5;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
B = A(mask);
% EXPECT: mask = matrix[3 x 3]
% EXPECT: B = matrix[None x 1]
