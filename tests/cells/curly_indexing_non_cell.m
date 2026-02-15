% Test: Curly indexing on non-cell
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]
% EXPECT: x = unknown

A = zeros(3, 3);
x = A{1};
