% Test: calling a function with fewer LHS targets than it returns is valid
% MATLAB allows [a, b] = f() when f returns 3 values — third is discarded

function [a, b, c] = helper()
    a = 1;
    b = 2;
    c = 3;
end

[x, y] = helper();
% No warning expected: requesting fewer outputs than declared is valid MATLAB
% EXPECT: x: scalar
% EXPECT: y: scalar
