% Test: calling a non-existent method falls back gracefully (UnknownShape, no crash)
% Field-not-found warning is expected since the method is not in the class.
% EXPECT: warnings = 1
% EXPECT: result = unknown

classdef Widget
  properties
    x
  end
  methods
    function obj = Widget(v)
      obj.x = v;
    end
  end
end

w = Widget(1.0);
result = w.nonexistent_method(42);
