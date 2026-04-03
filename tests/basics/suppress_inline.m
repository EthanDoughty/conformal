% Test: inline suppression of W_UNKNOWN_FUNCTION
% EXPECT: warnings = 0

x = unknown_func(ones(3,3));  % conformal:disable W_UNKNOWN_FUNCTION
