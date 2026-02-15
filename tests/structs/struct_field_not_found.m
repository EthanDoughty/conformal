% Test: Access non-existent field
% EXPECT: warnings = 1
% EXPECT: s = struct{x: scalar}
% EXPECT: r = unknown

s.x = 5;
r = s.y;
