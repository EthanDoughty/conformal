% EXPECT: warnings = 0
A = zeros(5, 5);
if cond
    x = 1;
else
    x = 8;
end
% x in [1, 8] after join
if 5 >= x
    % Flipped: 5 >= x means x <= 5. Guard is [-inf, 5].
    % meet([1, 8], [-inf, 5]) = [1, 5], within bounds.
    y = A(x, 1);
end
% EXPECT: y = scalar
