% Test: single-subscript range indexing folds the range length instead of
% collapsing to scalar. A vector base keeps its orientation, a 2D base takes
% the index's row orientation, End resolves against numel, and A(:) flattens
% to a column vector.
%
% EXPECT: x = matrix[1 x 10]
% EXPECT: y = matrix[1 x 4]
% EXPECT: c = matrix[10 x 1]
% EXPECT: d = matrix[3 x 1]
% EXPECT: A = matrix[3 x 4]
% EXPECT: v = matrix[12 x 1]
% EXPECT: g = matrix[1 x 4]
% EXPECT: h = matrix[1 x 9]
% EXPECT: s = matrix[1 x 5]
% EXPECT: warnings = 0

x = ones(1, 10);
y = x(2:5);

c = zeros(10, 1);
d = c(4:6);

A = zeros(3, 4);
v = A(:);
g = A(1:4);

h = x(2:end);
s = x(1:2:9);
