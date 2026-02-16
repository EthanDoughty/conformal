% Test end arithmetic in array indexing
A = [1 2 3; 4 5 6; 7 8 9];

% Last row, second-to-last column
x = A(end, end-1);
% EXPECT: x = scalar

% Second-to-last row, all columns
B = A(end-1, :);
% EXPECT: B = matrix[1 x 3]

% All rows, last two columns
C = A(:, end-1:end);
% EXPECT: C = matrix[3 x 2]
