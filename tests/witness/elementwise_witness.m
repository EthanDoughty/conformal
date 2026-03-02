% Elementwise mismatch — concrete witness for column conflict.
A = zeros(3, 4);
B = zeros(3, 5);
C = A .* B;  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH
% EXPECT: warnings = 1
