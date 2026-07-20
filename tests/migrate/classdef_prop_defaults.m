classdef Config
  properties
    retries = 3
    label = 'default'
    gain
    grid = zeros(2, 2)
  end
  methods
    function r = report(obj)
      r = obj.retries;
    end
  end
end
