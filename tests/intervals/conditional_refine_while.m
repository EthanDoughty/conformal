% EXPECT: warnings = 0
A = zeros(5, 5);
if cond
    x = 1;
else
    x = 8;
end
% x in [1, 8] after join
while x <= 5
    % Refinement: x narrowed from [1, 8] to [1, 5] by meet([1,8], [-inf, 5])
    y = A(x, 1);
    x = x + 1;
end
% EXPECT: y = scalar
