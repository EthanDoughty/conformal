% EXPECT: warnings = 1
a = 3;
b = a + 2;
A = zeros(b, b);
x = A(6, 1);  % EXPECT_WARNING: W_INDEX_OUT_OF_BOUNDS
% EXPECT: A = matrix[5 x 5]
