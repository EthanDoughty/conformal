% Matrix literal spacing disambiguation
% MATLAB rule: space-before + no-space-after on +/- means unary sign (new element)

% Two elements: 1, -2
A = [1 -2];
% EXPECT: A = matrix[1 x 2]

% One element: subtraction 1-2 wraps in 1x1 matrix
B = [1-2];
% EXPECT: B = matrix[1 x 1]

% One element: subtraction with spaces, still one element
C = [1 - 2];
% EXPECT: C = matrix[1 x 1]

% Three elements: 1, -2, 3
D = [1 -2 3];
% EXPECT: D = matrix[1 x 3]

% 2x2 matrix: two rows, each with two elements
E = [1 -2; 3 -4];
% EXPECT: E = matrix[2 x 2]

% Two elements with unary plus: 1, +2
F = [1 +2];
% EXPECT: F = matrix[1 x 2]

% Two matrix elements: x, -y (both 1x3)
x = ones(1, 3);
y = ones(1, 3);
G = [x -y];
% EXPECT: G = matrix[1 x 6]
% EXPECT: warnings = 0
