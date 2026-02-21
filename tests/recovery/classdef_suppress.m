% Test: classdef blocks are consumed without spurious W_END_OUTSIDE_INDEXING
% EXPECT: warnings = 0

x = 5;

classdef (Sealed) MyClass < handle
    properties (SetAccess = private)
        Name
        Value
    end

    events
        DataChanged
    end

    enumeration
        ModeA (1)
        ModeB (2)
    end

    methods
        function obj = MyClass(n, v)
            obj.Name = n;
            obj.Value = v;
        end

        function result = compute(obj)
            if obj.Value == 0
                result = 0;
                return;
            end
            result = obj.Value * 2;
        end
    end
end

y = x + 1;
% EXPECT: y = scalar
