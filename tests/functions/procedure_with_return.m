% Test: Procedure with explicit return statement
% No output vars, return just exits early
% EXPECT: warnings = 1
% EXPECT: A = unknown

function myproc(x)
    y = x * x;
    return;
end

A = myproc(zeros(3, 3));
