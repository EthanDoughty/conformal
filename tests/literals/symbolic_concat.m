% Test 15: Symbolic concatenation in matrix literals, no warnings are expected
% A is n x k and B is n x m.
% C = [A B] should infer n x (k+m).
% D = [A; A] should infer (n+n) x k.
%
% EXPECT: warnings = 0
% EXPECT: n = scalar
% EXPECT: k = scalar
% EXPECT: m = scalar
% EXPECT: A = matrix[n x k]
% EXPECT: B = matrix[n x m]
% EXPECT: C = matrix[n x (k+m)]
% EXPECT: D = matrix[(2*n) x k]

n = 4;
k = 5;
m = 2;

A = zeros(n, k);   % n x k
B = zeros(n, m);   % n x m

C = [A B];
D = [A; A];
