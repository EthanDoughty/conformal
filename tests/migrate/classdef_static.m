classdef Counter
  properties
    total
  end
  methods
    function obj = Counter(t)
      obj.total = t;
    end
    function s = scaled(obj, k)
      s = Counter.combine(obj.total, k);
    end
  end
  methods (Static)
    function c = combine(a, b)
      c = a * b;
    end
  end
end
