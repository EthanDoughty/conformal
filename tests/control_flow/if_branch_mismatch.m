% Test 10: Invalid symbolic mismatch inside control-flow, warning is expected
% A is n x k and B is n x n.
% A + B has incompatible dimensions (k vs n), so the analysis should detect it
% v0.9.3: Precision regression — then-branch error (unknown) now propagates through join
% (join_shape treats unknown as absorbing top). More sound, less precise.

% EXPECT: warnings = 1
% EXPECT: A = matrix[n x k]
% EXPECT: B = matrix[n x n]
% EXPECT: C = unknown
% EXPECT: k = scalar
% EXPECT: n = scalar

n = 4;
k = 5;

A = zeros(n, k);   % n x k
B = zeros(n, n);   % n x n

if n > 0
    C = A + B;
else
    C = A;
end
