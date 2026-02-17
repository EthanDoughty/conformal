% EXPECT: warnings = 0
a = 10;
b = 5;
c = a / b;
d = a ./ b;
% EXPECT: c = scalar
% EXPECT: d = scalar
