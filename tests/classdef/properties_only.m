% Test: classdef with only a properties block, no constructor method.
% Without a constructor body, properties are unknown shape.
% EXPECT: warnings = 0
% EXPECT: obj = struct{x: unknown, y: unknown}

classdef Point
  properties
    x
    y
  end
end

obj = Point();
