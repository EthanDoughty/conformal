% EXPECT: warnings = 0
A = zeros(10, 10);
if cond
    x = 1;
else
    x = 15;
end
% After join: x in [1, 15]. A(x,1) would warn "may exceed dim 10".
if x > 0 && x <= 10
    y = A(x, 1);
end
% EXPECT: y = scalar
