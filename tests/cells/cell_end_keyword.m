% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 3]
% EXPECT: x = string

c = {1, 2, 'last'};
x = c{end};
