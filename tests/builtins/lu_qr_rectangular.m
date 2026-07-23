% Test: lu and qr rectangular factor sizes (min(m,n) clamping)
% EXPECT: warnings = 0

% === lu: tall matrix (LAPACK dgetrf: L m-by-k, U k-by-n, k = min(m,n)) ===
A = zeros(6, 3);
[L2, U2] = lu(A);
% EXPECT: L2 = matrix[6 x 3]
% EXPECT: U2 = matrix[3 x 3]
[L3, U3, P3] = lu(A);
% EXPECT: L3 = matrix[6 x 3]
% EXPECT: U3 = matrix[3 x 3]
% EXPECT: P3 = matrix[6 x 6]

% === lu: wide matrix ===
B = zeros(3, 6);
[L2b, U2b] = lu(B);
% EXPECT: L2b = matrix[3 x 3]
% EXPECT: U2b = matrix[3 x 6]
[L3b, U3b, P3b] = lu(B);
% EXPECT: L3b = matrix[3 x 3]
% EXPECT: U3b = matrix[3 x 6]
% EXPECT: P3b = matrix[3 x 3]

% === lu: square matrix (uniform rule must not disturb the working case) ===
C = zeros(4, 4);
[L2c, U2c] = lu(C);
% EXPECT: L2c = matrix[4 x 4]
% EXPECT: U2c = matrix[4 x 4]
[L3c, U3c, P3c] = lu(C);
% EXPECT: L3c = matrix[4 x 4]
% EXPECT: U3c = matrix[4 x 4]
% EXPECT: P3c = matrix[4 x 4]

% === qr: full (no flag) unchanged ===
D = zeros(10, 3);
[Q, R] = qr(D);
% EXPECT: Q = matrix[10 x 10]
% EXPECT: R = matrix[10 x 3]

% === qr: economy flag, three spellings ===
[Qe0, Re0] = qr(D, 0);
% EXPECT: Qe0 = matrix[10 x 3]
% EXPECT: Re0 = matrix[3 x 3]
[Qe1, Re1] = qr(D, 'econ');
% EXPECT: Qe1 = matrix[10 x 3]
% EXPECT: Re1 = matrix[3 x 3]
[Qe2, Re2] = qr(D, "econ");
% EXPECT: Qe2 = matrix[10 x 3]
% EXPECT: Re2 = matrix[3 x 3]

% === qr: wide matrix, economy coincides with full since m <= n ===
E = zeros(3, 10);
[Qw, Rw] = qr(E, 0);
% EXPECT: Qw = matrix[3 x 3]
% EXPECT: Rw = matrix[3 x 10]
