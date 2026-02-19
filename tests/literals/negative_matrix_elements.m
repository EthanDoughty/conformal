% Test: matrix literals with negative elements
% Bug B1: [-1 -2] was parsed as (-1) - 2 = -3 (scalar) instead of [-1, -2] (1x2)

% EXPECT: warnings = 0
% EXPECT: a = matrix[1 x 2]
% EXPECT: b = matrix[3 x 2]
% EXPECT: c = matrix[1 x 3]
% EXPECT: d = matrix[1 x 2]
% EXPECT: e = matrix[2 x 2]

a = [-1 -2];
b = [-5 -1; -1 -1; 4 -1];
c = [3 -1 -2];
d = [-1, -2];
e = [-1 -2; -3 -4];
