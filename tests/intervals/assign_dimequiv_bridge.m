% Direct test: k assigned exact value via constraint, bridge propagates.
% A*B requires inner dims to match: A is n x 4, B is k x 3, so k == 4.
% EXPECT: warnings = 0
% EXPECT: C = matrix[4 x 4]
A = rand(n, 4);
B = rand(k, 3);
D = A * B;
C = zeros(k, k);
