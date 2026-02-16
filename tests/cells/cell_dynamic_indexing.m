% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 2]
% EXPECT: i = scalar
% EXPECT: x = unknown

c = {1, zeros(2, 2)};
i = 1;
x = c{i};
