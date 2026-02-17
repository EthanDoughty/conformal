% EXPECT: warnings = 0
A = zeros(5, 5);
if cond
    x = 2;
else
    x = 7;
end
% x in [2, 7] after join
if x <= 5
    % x narrowed to [2, 5] by meet([2,7], [-inf, 5])
    y = A(x, 1);
end
% Without conditional refinement, A(x, 1) inside the guard would still see [2, 7] and warn.
% With refinement, [2, 5] is within bounds -> no warning.
% EXPECT: y = scalar
