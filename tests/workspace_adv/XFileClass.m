classdef XFileClass
  properties
    A
  end
  methods
    function obj = XFileClass(m)
      obj.A = m;
    end
    function y = apply(obj, x)
      y = obj.A * x;
    end
  end
end
