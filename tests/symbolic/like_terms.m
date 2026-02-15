% Test: Like terms collected in symbolic dimensions
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x n]
% EXPECT: D = matrix[(2*n) x n]

n = 5;
A = zeros(n, n);
D = [A; A];
