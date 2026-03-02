% EXPECT: warnings = 1
A = zeros(3, 4);
x = A(5, 1);  % EXPECT_WARNING: W_INDEX_OUT_OF_BOUNDS
% EXPECT: x = scalar
