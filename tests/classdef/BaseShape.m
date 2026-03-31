% SKIP_TEST
classdef BaseShape
    properties
        color
        lineWidth
    end
    methods
        function obj = BaseShape(c, lw)
            obj.color = c;
            obj.lineWidth = lw;
        end
        function a = area(obj)
            a = 0;
        end
    end
end
