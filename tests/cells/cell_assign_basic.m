% Test: Cell element assignment (1D)
% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 3]
% EXPECT: x = scalar

c = {1, 2, 3};
c{2} = 42;
x = c{2};
