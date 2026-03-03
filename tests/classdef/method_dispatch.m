% Test: classdef method dispatch - obj.method(args) routes to method(obj, args)
% EXPECT: warnings = 0
% EXPECT: result = scalar

classdef Counter
  properties
    value
  end
  methods
    function obj = Counter(v)
      obj.value = v;
    end
    function y = get_value(obj)
      y = obj.value;
    end
  end
end

c = Counter(5.0);
result = c.get_value();
