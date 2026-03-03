classdef MyVehicle
  properties
    speed
    capacity
  end
  methods
    function obj = MyVehicle(s, c)
      obj.speed    = s;
      obj.capacity = c;
    end
    function y = get_speed(obj)
      y = obj.speed;
    end
  end
end
