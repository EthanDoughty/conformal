% Test: classdef blocks are consumed without spurious W_END_OUTSIDE_INDEXING
% EXPECT: warnings = 0

x = 5;

classdef MyClass
    properties
        Name
        Value
    end

    methods
        function obj = MyClass(n, v)
            obj.Name = n;
            obj.Value = v;
        end

        function result = compute(obj)
            result = obj.Value * 2;
        end
    end
end

y = x + 1;
% EXPECT: y = scalar
