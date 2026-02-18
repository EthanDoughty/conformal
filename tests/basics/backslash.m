% Test: backslash mldivide operator A\b

% EXPECT: warnings = 1
% EXPECT: s = scalar
% EXPECT: x = matrix[3 x 1]
% EXPECT: y = matrix[4 x 1]
% EXPECT: z = unknown

% scalar \ scalar -> scalar
s = 2 \ 3;

% A[m x n] \ b[m x p] -> [n x p]
A = zeros(4, 3);
b = zeros(4, 1);
x = A \ b;

% scalar \ matrix -> matrix
y = 2 \ zeros(4, 1);

% row mismatch -> warning + unknown
c = zeros(5, 1);
z = A \ c;
