% EXPECT: warnings = 1
a = 3;
b = a + 2;
A = zeros(b, b);
x = A(6, 1);
% EXPECT: A = matrix[b x b]
