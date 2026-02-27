% Test: classdef with properties that have default values.
% Property names should be extracted even when default values are present.
% EXPECT: warnings = 0
% EXPECT: obj = struct{x: unknown, y: unknown}

classdef Widget
  properties
    x = 0
    y = []
  end
end

obj = Widget();
