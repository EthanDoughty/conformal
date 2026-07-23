% Test: conjugate-gradient style step
% alpha = (r' * r) / (p' * Ap); x = x + alpha * p

% EXPECT: warnings = 0
% EXPECT: alpha = matrix[1 x 1]
% EXPECT: x = matrix[5 x 1]

r = zeros(5, 1);
p = zeros(5, 1);
Ap = zeros(5, 1);
x = zeros(5, 1);

alpha = (r' * r) / (p' * Ap);
x = x + alpha * p;
