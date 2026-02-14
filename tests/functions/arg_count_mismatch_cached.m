% Test: Function called with wrong number of arguments
% Arg count mismatch → warning, return unknown, no cache interaction
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = two_args(x, z)
    y = x + z;
end

A = two_args(zeros(3, 3));  % Missing 2nd arg
