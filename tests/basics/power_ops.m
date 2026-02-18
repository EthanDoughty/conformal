% Test: ^ and .^ power operators

% EXPECT: warnings = 1
% EXPECT: s = scalar
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT: C = matrix[3 x 3]
% EXPECT: v = matrix[1 x 4]
% EXPECT: w = matrix[1 x 4]
% EXPECT: bad = unknown

% scalar ^ scalar -> scalar
s = 2 ^ 3;

% square matrix ^ scalar -> same shape
A = zeros(3, 3);
B = A ^ 2;

% element-wise .^ (broadcast like .*)
C = A .^ 2;

% element-wise .^ on vectors
v = zeros(1, 4);
w = v .^ 2;

% non-square matrix ^ scalar -> warning + unknown
bad = zeros(2, 3) ^ 2;
