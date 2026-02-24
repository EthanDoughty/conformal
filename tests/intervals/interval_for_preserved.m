% EXPECT: warnings = 0
% The for-loop iteration variable must not be widened by the interval
% widening pass -- its interval is set by the range declaration.
A = zeros(5, 5);
for i = 1:5
    x = A(i, 1);
end
% EXPECT: x = scalar
