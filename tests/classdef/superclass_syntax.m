% Test: classdef Foo < Bar syntax parses without error.
% Superclass stored but not resolved. Properties extracted correctly.
% EXPECT: warnings = 0
% EXPECT: obj = struct{value: unknown}

classdef Counter < handle
  properties
    value
  end
end

obj = Counter();
