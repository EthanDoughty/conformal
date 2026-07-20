classdef Sensor
  properties
    Voltage (1,1) double {mustBeReal} = 3.3
    Name char = 'probe'
    Offset (1,1) double
  end
  methods
    function v = read(obj)
      v = obj.Voltage + obj.Offset;
    end
  end
end
