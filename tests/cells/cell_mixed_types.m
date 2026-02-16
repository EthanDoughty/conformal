% Test: Cell with mixed element types
% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 3]
% EXPECT: a = matrix[2 x 2]
% EXPECT: b = string
% EXPECT: s = scalar

c = {zeros(2, 2), 'hello', 42};
a = c{1};
b = c{2};
s = c{3};
