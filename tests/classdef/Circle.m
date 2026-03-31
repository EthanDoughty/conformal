% SKIP_TEST
classdef Circle < BaseShape
    properties
        radius
    end
    methods
        function obj = Circle(r, c)
            obj.radius = r;
            obj.color = c;
            obj.lineWidth = 1;
        end
        function a = area(obj)
            a = 3.14159 * obj.radius^2;
        end
    end
end
