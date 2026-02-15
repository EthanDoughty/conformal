% Test arithmetic dimension expressions in builtin constructors
% EXPECT: warnings = 0

% Simple addition
A = zeros(n+1, m);
% EXPECT: A = matrix[(n+1) x m]

% Simple multiplication
B = zeros(2*n, k);
% EXPECT: B = matrix[(2*n) x k]

% Nested addition
C = zeros(n+m+k, 1);
% EXPECT: C = matrix[(k+m+n) x 1]

% Subtraction
D = zeros(n-1, m);
% EXPECT: D = matrix[(n-1) x m]

% Mixed symbolic and concrete
E = zeros(n+2, 3*m);
% EXPECT: E = matrix[(n+2) x (3*m)]

% reshape with arithmetic dimensions
F = reshape(A, 2*n, m+1);
% EXPECT: F = matrix[(2*n) x (m+1)]

% Edge case: nested function call returns None
G = zeros(length(A), m);
% EXPECT: G = matrix[None x m]

% Concrete arithmetic (folded to constants)
H = zeros(2+3, 4*2);
% EXPECT: H = matrix[5 x 8]

% Nested multiplication
I = zeros(2*3*n, k);
% EXPECT: I = matrix[(6*n) x k]
