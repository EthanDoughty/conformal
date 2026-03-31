% Test: child without method override inherits parent method
% EXPECT: warnings = 0

s = Square(4, 'blue');
a = s.area();
