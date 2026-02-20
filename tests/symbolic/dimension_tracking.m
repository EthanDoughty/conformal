% Test 5: Valid colon vector and transpose, no warnings are expected
% v = 1:n is a 1xn row vector, so v' is nx1
% A * v' is valid when A is nxn

% EXPECT: warnings = 0
% EXPECT: A = matrix[5 x 5]
% EXPECT: n = scalar
% EXPECT: v = matrix[1 x None]
% EXPECT: y = matrix[5 x 1]

n = 5;
v = 1:n;
A = ones(n, n);
y = A * v';
