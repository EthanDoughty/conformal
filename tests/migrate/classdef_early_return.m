classdef G
  methods
    function obj = G(x)
      if x < 0
        obj.v = 0;
        return
      end
      obj.v = x;
    end
  end
end
