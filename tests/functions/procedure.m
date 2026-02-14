% Test: Procedure-style function (no return value)
% EXPECT: warnings = 1
% EXPECT: A = unknown

function myproc(x)
    y = x * x;
end

A = myproc(zeros(3, 3));
