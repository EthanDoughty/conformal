% Test: when the class constructor is in classRegistry.methods and is invoked
% via EvalExpr FieldAccess dispatch (which prepends obj), the effective
% args.Length becomes parms.Length + 1. The self-offset prevents a false
% W_FUNCTION_ARG_COUNT_MISMATCH in makeClassConstructorShape.
% EXPECT: warnings = 0

classdef Box
  properties
    w
    h
  end
  methods
    function obj = Box(w, h)
      obj.w = w;
      obj.h = h;
    end
    function a = area(obj)
      a = obj.w * obj.h;
    end
  end
end

b = Box(3, 4);
a = b.area();
