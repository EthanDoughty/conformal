% EXPECT: warnings = 1
A = zeros(3, 4);
x = A(5, 1);
% EXPECT: x = scalar
