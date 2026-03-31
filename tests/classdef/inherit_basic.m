% Test: child class inherits parent properties and methods
% EXPECT: warnings = 0
% EXPECT: c = struct{color: string, lineWidth: scalar, radius: scalar}

c = Circle(5, 'red');
clr = c.color;
lw = c.lineWidth;
r = c.radius;
a = c.area();
