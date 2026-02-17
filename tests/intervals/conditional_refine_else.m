% EXPECT: warnings = 0
A = zeros(5, 5);
if cond
    x = 3;
else
    x = 8;
end
% x in [3, 8] after join. A(x,1) would normally warn "may exceed dim 5".
if x > 5
    y = 1;
else
    % else-body: negation of (x > 5) gives (x <= 5), so guard is [-inf, 5].
    % meet([3, 8], [-inf, 5]) = [3, 5], which is within bounds.
    z = A(x, 1);
end
% EXPECT: z = scalar
