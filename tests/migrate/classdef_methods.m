classdef Vec
  properties
    data
  end
  methods
    function obj = Vec(d)
      obj.data = d;
    end
    function n = norm(obj)
      n = sqrt(obj.dot(obj));
    end
    function s = dot(obj, other)
      s = sum(obj.data .* other.data);
    end
  end
end
