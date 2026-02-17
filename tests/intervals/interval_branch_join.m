% EXPECT: warnings = 0
if cond
    x = 3;
else
    x = 7;
end
y = 10 / x;
% EXPECT: y = scalar
