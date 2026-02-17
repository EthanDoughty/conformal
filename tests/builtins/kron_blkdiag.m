% Test: kron and blkdiag builtin shape rules
% EXPECT: warnings = 0

% --- kron tests ---

% Concrete: kron([2x3], [4x5]) -> [8 x 15]
A = zeros(2, 3);
B = zeros(4, 5);
K1 = kron(A, B);
% EXPECT: K1 = matrix[8 x 15]

% Symbolic: kron([n x m], [p x q]) -> [(n*p) x (m*q)]
C = zeros(n, m);
D = zeros(p, q);
K2 = kron(C, D);
% EXPECT: K2 = matrix[(n*p) x (m*q)]

% Scalar first arg: kron(scalar, [3x3]) -> [3 x 3]
K3 = kron(5, zeros(3, 3));
% EXPECT: K3 = matrix[3 x 3]

% Scalar second arg: kron([2x4], scalar) -> [2 x 4]
K4 = kron(zeros(2, 4), 7);
% EXPECT: K4 = matrix[2 x 4]

% Mixed symbolic/concrete: kron([2 x n], [3 x m]) -> [6 x (m*n)]
K5 = kron(zeros(2, n), zeros(3, m));
% EXPECT: K5 = matrix[6 x (m*n)]

% --- blkdiag tests ---

% Concrete 2-arg: blkdiag([2x3], [4x5]) -> [6 x 8]
BD1 = blkdiag(A, B);
% EXPECT: BD1 = matrix[6 x 8]

% Symbolic 2-arg: blkdiag([n x m], [p x q]) -> [(n+p) x (m+q)]
BD2 = blkdiag(C, D);
% EXPECT: BD2 = matrix[(n+p) x (m+q)]

% Concrete 3-arg: blkdiag([2x3], [4x5], [1x1]) -> [7 x 9]
BD3 = blkdiag(A, B, ones(1, 1));
% EXPECT: BD3 = matrix[7 x 9]

% Scalar arg: blkdiag(scalar, [3x3]) -> [4 x 4]
BD4 = blkdiag(5, zeros(3, 3));
% EXPECT: BD4 = matrix[4 x 4]

% Mixed symbolic/concrete: blkdiag([2 x 3], [n x m]) -> [(n+2) x (m+3)]
BD5 = blkdiag(A, C);
% EXPECT: BD5 = matrix[(n+2) x (m+3)]

% Single arg: blkdiag([2x3]) -> [2 x 3]
BD6 = blkdiag(A);
% EXPECT: BD6 = matrix[2 x 3]
