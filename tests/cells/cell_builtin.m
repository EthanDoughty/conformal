% Test: cell() builtin constructor
% EXPECT: warnings = 0
% EXPECT: c1 = cell[3 x 4]
% EXPECT: c2 = cell[5 x 6]

c1 = cell(3, 4);
n = 5;
m = 6;
c2 = cell(n, m);
