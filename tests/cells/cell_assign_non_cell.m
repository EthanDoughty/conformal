% Test: Cell assignment on non-cell
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]

A = zeros(3, 3);
A{1, 2} = 5;
