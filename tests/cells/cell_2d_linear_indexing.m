% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 2]
% EXPECT: a = scalar
% EXPECT: b = matrix[3 x 1]
% EXPECT: x = scalar

c = {1, zeros(3, 1); 'test', 99};
a = c{1};
b = c{3};
x = c{4};
