% Test: indexed assignment into struct field
% EXPECT: s = struct{x: matrix[1 x 3], y: scalar}
% EXPECT: warnings = 0

s.x = [1 2 3];
s.y = 42;
s.x(1) = 10;
s.x(2) = 20;
