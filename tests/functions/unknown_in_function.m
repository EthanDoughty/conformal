% Test: Function calls unknown builtin, propagates unknown to caller
% Dual-location warning: "line 3 (in bad_func, called from line 6)"
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = bad_func(x)
    y = unknown_builtin(x);
end

A = bad_func(zeros(3, 3));
