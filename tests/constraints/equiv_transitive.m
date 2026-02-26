% Test: transitive concrete propagation through equivalence chain.
% A*B records A.cols==B.rows => 5 == m. B+D records B.rows==D.rows => m == k.
% Transitive: k is in the same class as 5 via m. Post-analysis resolves k -> 5.
% EXPECT: warnings = 0
% EXPECT: G = matrix[5 x 3]
A = rand(n, 5);
B = rand(m, p);
C = A * B;
D = rand(k, p);
E = B + D;
G = zeros(k, 3);
