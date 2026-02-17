% EXPECT: warnings = 0
A = zeros(5, 5);
if cond
    x = 3;
else
    x = 8;
end
% After join: x in [3, 8]. Without guard, A(x,1) would warn "may exceed dim 5".
if x <= 5
    y = A(x, 1);
end
% EXPECT: y = scalar
