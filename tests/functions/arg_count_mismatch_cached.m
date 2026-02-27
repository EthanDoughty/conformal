% Test: Function called with fewer arguments than declared (nargin support)
% Too few args: no warning (optional args). Result is unknown because z is unbound.
% EXPECT: warnings = 0
% EXPECT: A = unknown

function y = two_args(x, z)
    y = x + z;
end

A = two_args(zeros(3, 3));  % Missing 2nd arg -- nargin=1, z is Bottom
