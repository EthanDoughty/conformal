% Test: length() and numel() interval propagation.
% Verifies that length/numel inject concrete intervals into valueRanges
% and symbolic dim aliases for vectors.
% EXPECT: warnings = 0
% EXPECT: n1 = scalar
% EXPECT: n2 = scalar
% EXPECT: n3 = scalar
% EXPECT: n4 = scalar
% EXPECT: idx = matrix[1 x n]

% length of known-size column vector: interval [10, 10]
v = zeros(10, 1);
n1 = length(v);

% length of known-size matrix: max(3, 5) = 5
M = zeros(3, 5);
n2 = length(M);

% numel of known-size matrix: 3*4 = 12
N = zeros(3, 4);
n3 = numel(N);

% length of scalar: 1
s = 42;
n4 = length(s);

% Downstream indexing: length-bounded loop should not produce OOB
w = zeros(8, 1);
k = length(w);
for i = 1:k
    w(i);
end

% Symbolic aliasing: length(vector) aliases to the symbolic dim
v2 = zeros(n, 1);
k2 = length(v2);
idx = 1:k2;  % should chain: k2 aliases n, so 1:k2 = matrix[1 x n]
