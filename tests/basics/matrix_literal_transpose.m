% Test: transpose of matrix literal
% EXPECT: x = matrix[2 x 2]
% EXPECT: y = matrix[3 x 1]
% EXPECT: z = matrix[1 x 3]
% EXPECT: warnings = 0

A = [1 2; 3 4];
x = A';
y = [1 2 3]';
z = [1; 2; 3]';
