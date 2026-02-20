% Test 15: Symbolic concatenation in matrix literals, no warnings are expected
% A is n x k and B is n x m.
% C = [A B] should infer n x (k+m).
% D = [A; A] should infer (n+n) x k.
%
% EXPECT: warnings = 0
% EXPECT: n = scalar
% EXPECT: k = scalar
% EXPECT: m = scalar
% EXPECT: A = matrix[4 x 5]
% EXPECT: B = matrix[4 x 2]
% EXPECT: C = matrix[4 x 7]
% EXPECT: D = matrix[8 x 5]

n = 4;
k = 5;
m = 2;

A = zeros(n, k);   % n x k
B = zeros(n, m);   % n x m

C = [A B];
D = [A; A];
