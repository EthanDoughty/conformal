% Test: reshape and repmat builtin shape rules
% EXPECT: warnings = 0
% EXPECT: reshape_concrete = matrix[3 x 2]
% EXPECT: reshape_symbolic = matrix[n x m]
% EXPECT: repmat_scalar = matrix[2 x 3]
% EXPECT: repmat_concrete = matrix[4 x 6]
% EXPECT: repmat_symbolic = matrix[(n*k) x (m*k)]
% EXPECT: reshape_from_scalar = matrix[3 x 2]
% EXPECT: reshape_nonmatching = matrix[4 x 4]

% Test 1: reshape with concrete dimensions
A = zeros(2, 3);
reshape_concrete = reshape(A, 3, 2);

% Test 2: reshape with symbolic dimensions
n = 5;
m = 4;
B = ones(6, 1);
reshape_symbolic = reshape(B, n, m);

% Test 3: repmat with scalar input and concrete replication factors
repmat_scalar = repmat(5, 2, 3);

% Test 4: repmat with matrix input and concrete replication factors
C = zeros(2, 3);
repmat_concrete = repmat(C, 2, 2);

% Test 5: repmat with symbolic matrix input and symbolic replication factors
n = 2;
m = 3;
k = 4;
D = ones(n, m);
repmat_symbolic = repmat(D, k, k);

% Test 6: reshape from scalar (validates "trust dimensions" for scalars)
reshape_from_scalar = reshape(5, 3, 2);

% Test 7: reshape with non-matching element count (no warning expected, returns matrix[4 x 4])
reshape_nonmatching = reshape(zeros(2, 3), 4, 4);
