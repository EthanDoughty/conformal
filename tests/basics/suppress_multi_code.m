% Test: multiple codes in one directive
% EXPECT: warnings = 0

x = unknown_func(ones(3,3));  % conformal:disable W_UNKNOWN_FUNCTION W_STRUCT_FIELD_NOT_FOUND
