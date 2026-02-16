% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 4]
% EXPECT: x = unknown

c = {1, 2, 3, 'last'};
x = c{1:end};
