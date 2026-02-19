% Test: global and persistent declarations are parsed and targets are havocked
% EXPECT: warnings = 0
% EXPECT: x = unknown
% EXPECT: y = unknown

global x y z
persistent a b

x = x + 1;
y = a;
