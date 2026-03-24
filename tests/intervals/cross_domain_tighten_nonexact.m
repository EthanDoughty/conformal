% Negative test: non-exact interval must NOT propagate a concrete dim.
% A has unknown shape (symbolic), so n = size(A,1) stays symbolic.
% zeros(n,3) should have symbolic rows, not concrete.
% EXPECT: warnings = 0
% EXPECT: B = matrix[n x 3]
A = rand(n, 3);
k = size(A, 1);
B = zeros(k, 3);
