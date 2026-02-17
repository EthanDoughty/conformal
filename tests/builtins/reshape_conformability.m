% Test: reshape conformability checking
% EXPECT: warnings = 2

% Concrete match: 2*3 = 3*2, no warning
A = zeros(2, 3);
B = reshape(A, 3, 2);
% EXPECT: B = matrix[3 x 2]

% Concrete mismatch: 2*3=6 != 4*4=16, warning
C = reshape(A, 4, 4);
% EXPECT: C = matrix[4 x 4]

% Scalar input: 1 != 3*2=6, warning
D = reshape(5, 3, 2);
% EXPECT: D = matrix[3 x 2]

% Symbolic input: can't prove mismatch, no warning
E = zeros(n, m);
F = reshape(E, 3, 2);
% EXPECT: F = matrix[3 x 2]

% Symbolic output: can't prove mismatch, no warning
G = reshape(A, n, m);
% EXPECT: G = matrix[n x m]

% Both symbolic: can't prove mismatch, no warning
H = reshape(E, p, q);
% EXPECT: H = matrix[p x q]

% Concrete match: 4*3=12 = 6*2=12, no warning
I = zeros(4, 3);
J = reshape(I, 6, 2);
% EXPECT: J = matrix[6 x 2]
