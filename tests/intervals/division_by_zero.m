% EXPECT: warnings = 1
a = 10;
b = 0;
c = a / b;  % EXPECT_WARNING: W_DIVISION_BY_ZERO
% EXPECT: c = scalar
