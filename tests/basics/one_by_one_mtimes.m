% Test: a 1x1 matrix value (e.g. a dot product) is treated as a scalar in *.
% v' * v (a dot product, 1x5 times 5x1) must still produce matrix[1 x 1]
% via the ordinary matmul rule, and multiplying that 1x1 result by a
% non-scalar matrix must not draw W_INNER_DIM_MISMATCH.

% EXPECT: warnings = 0
% EXPECT: alpha = matrix[1 x 1]
% EXPECT: y = matrix[5 x 1]

p = zeros(5, 1);
alpha = p' * p;
y = alpha * p;
