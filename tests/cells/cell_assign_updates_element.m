% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 2]
% EXPECT: x = matrix[4 x 4]

c = {1, 2};
c{1} = ones(4, 4);
x = c{1};
