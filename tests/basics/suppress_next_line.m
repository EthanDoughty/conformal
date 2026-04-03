% Test: next-line suppression
% EXPECT: warnings = 0

% conformal:disable-next-line W_UNKNOWN_FUNCTION
x = unknown_func(ones(3,3));
