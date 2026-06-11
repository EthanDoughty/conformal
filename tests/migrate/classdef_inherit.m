classdef Circle < Shape
    properties
        radius
    end
    methods
        function obj = Circle(r)
            obj = obj@Shape('round');
            obj.radius = r;
        end
        function a = area(obj)
            a = pi * obj.radius^2;
        end
    end
end
