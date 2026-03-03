% Test: return inside function should NOT warn
% EXPECT: warnings = 0
function y = myFunc(x)
    y = x;
    return;
end
