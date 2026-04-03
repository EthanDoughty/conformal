% conformal:disable W_UNKNOWN_FUNCTION
% Test: file-level suppression
% EXPECT: warnings = 0

x = unknown_func(ones(3,3));
y = another_unknown(x);
