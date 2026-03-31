% SKIP_TEST
classdef HandleChild < handle
    properties
        value
    end
    methods
        function obj = HandleChild(v)
            obj.value = v;
        end
    end
end
