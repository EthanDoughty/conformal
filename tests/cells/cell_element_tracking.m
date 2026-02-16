% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 3]
% EXPECT: x = scalar
% EXPECT: y = matrix[2 x 3]
% EXPECT: z = string

c = {42, zeros(2, 3), 'hello'};
x = c{1};
y = c{2};
z = c{3};
