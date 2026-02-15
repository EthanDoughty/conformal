% Test: Cell range indexing with colon operator
% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 5]
% EXPECT: x = unknown
% EXPECT: y = unknown

c = {1, 2, 3, 4, 5};
x = c{1:3};
y = c{2:4};
