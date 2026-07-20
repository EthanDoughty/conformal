classdef MathKit
  methods (Static)
    function y = square(x)
      y = x * x;
    end
    function z = quad(x)
      z = MathKit.square(MathKit.square(x));
    end
  end
  methods (Static = false)
    function n = tag(obj)
      n = obj.label;
    end
  end
end
