classdef C < B
  methods
    function obj = C(x)
      obj = obj@B(x);
    end
    function y = go(obj)
      y = step@B(obj, 2);
    end
  end
end
