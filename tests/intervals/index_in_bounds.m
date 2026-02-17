% EXPECT: warnings = 0
A = zeros(3, 4);
x = A(2, 3);
% EXPECT: x = scalar
