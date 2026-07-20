classdef Widget
  properties
    width = 10
    height = 4
  end
  methods
    function obj = Widget(w)
      obj.width = w;
    end
    function a = area(obj)
      a = obj.width * obj.height;
    end
  end
end
