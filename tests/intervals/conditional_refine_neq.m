% EXPECT: warnings = 0
A = zeros(5, 5);
x = 3;
if x ~= 3
    % then-body: no refinement (can't exclude a point from interval)
    y = 1;
else
    % else-body: negation of ~= is ==, so x narrowed to [3, 3]
    z = A(x, 1);
end
% EXPECT: z = scalar
