% Test: diag of a matrix returns min(m,n)-by-1; vector forms are untouched
% EXPECT: warnings = 0

% Square matrix -> min(m,n) = m = n
d_square = diag(zeros(4, 4));
% EXPECT: d_square = matrix[4 x 1]

% Tall matrix -> min(m,n) = n
d_tall = diag(zeros(5, 3));
% EXPECT: d_tall = matrix[3 x 1]

% Wide matrix -> min(m,n) = m
d_wide = diag(zeros(3, 5));
% EXPECT: d_wide = matrix[3 x 1]

% Symbolic square matrix -> minDim's structural-equality arm (n = n)
d_sym_square = diag(zeros(n, n));
% EXPECT: d_sym_square = matrix[n x 1]

% Symbolic rectangular matrix -> minDim cannot decide, stays unknown
d_sym_rect = diag(zeros(n, m));
% EXPECT: d_sym_rect = matrix[None x 1]

% Downstream cascade: diag of a square matrix gives a column vector, and
% diag of that column vector rebuilds a square matrix of the same extent.
A = zeros(4, 4);
d = diag(A);
% EXPECT: d = matrix[4 x 1]
B = diag(d);
% EXPECT: B = matrix[4 x 4]
