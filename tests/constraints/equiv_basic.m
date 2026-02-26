% Test: basic dimension equivalence via matmul, vertcat benefits with 0 warnings.
% A*B records A.cols == B.rows (k == n constraint). [B; D] checks B.cols == D.cols.
% EXPECT: warnings = 0
% EXPECT: F = matrix[(n+5) x 5]
A = rand(3, k);
B = rand(n, 5);
C = A * B;
D = rand(5, 5);
F = [B; D];
