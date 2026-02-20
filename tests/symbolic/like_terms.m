% Test: Like terms collected in symbolic dimensions
% EXPECT: warnings = 0
% EXPECT: A = matrix[5 x 5]
% EXPECT: D = matrix[10 x 5]

n = 5;
A = zeros(n, n);
D = [A; A];
