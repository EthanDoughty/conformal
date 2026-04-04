% Test: classdef method called via obj.method(args) should not trigger
% W_FUNCTION_ARG_COUNT_MISMATCH even though EvalExpr prepends obj making
% args.Length == parms.Length. The self-offset suppresses false positives
% from the self/obj implicit parameter in MATLAB classdef methods.
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

classdef Processor
  methods
    function obj = Processor()
    end
    function y = process(obj, x)
      y = x * 2;
    end
  end
end

p = Processor();
result = p.process(ones(3, 3));
