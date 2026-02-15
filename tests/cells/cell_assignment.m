% Test: Cell variable assignment
% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 3]
% EXPECT: d = cell[2 x 3]

c = {1, 2, 3; 4, 5, 6};
d = c;
