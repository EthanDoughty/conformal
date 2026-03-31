% SKIP_TEST
classdef Square < BaseShape
    properties
        side
    end
    methods
        function obj = Square(s, c)
            obj.side = s;
            obj.color = c;
            obj.lineWidth = 1;
        end
    end
end
