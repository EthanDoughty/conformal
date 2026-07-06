% Block comment contents must not reach the parser: before block-comment
% support, the assignment inside the block executed and flipped A to 1x3.

A = zeros(3, 4);
%{
A = zeros(1, 3);
This prose line would otherwise parse as a statement.
%}
x = zeros(4, 1);
y = A * x;

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: y = matrix[3 x 1]
