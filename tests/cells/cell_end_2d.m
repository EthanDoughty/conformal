% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 2]
% EXPECT: x = scalar

c = {1, 2; 3, 4};
x = c{end, end};
