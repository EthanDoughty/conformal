% Test: right division mrdivide operator A/B

% EXPECT: warnings = 1
% EXPECT: s = scalar
% EXPECT: x = matrix[2 x 4]
% EXPECT: z = unknown
% EXPECT: w = matrix[3 x 3]

% scalar / scalar -> scalar
s = 6 / 2;

% A[m x k] / B[n x k] -> [m x n]
A = zeros(2, 3);
B = zeros(4, 3);
x = A / B;

% column mismatch -> warning + unknown
C = zeros(2, 4);
z = A / C;  % EXPECT_WARNING: W_MRDIVIDE_DIM_MISMATCH

% same shape square: D / D -> matrix[3 x 3]
D = zeros(3, 3);
w = D / D;
