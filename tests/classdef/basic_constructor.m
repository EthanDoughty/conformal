% Test: classdef with properties and constructor
% Constructor assigns scalar properties; obj should be a struct with both fields.
% EXPECT: warnings = 0
% EXPECT: result = struct{x: scalar, y: scalar}

classdef Foo
  properties
    x
    y
  end
  methods
    function obj = Foo(a, b)
      obj.x = a;
      obj.y = b;
    end
  end
end

result = Foo(3.0, 4.0);
