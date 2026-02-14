% Test: Return statement (early exit from function)
% EXPECT: warnings = 0
% EXPECT: A = scalar

function result = early_exit(x)
    result = x;
    return;
end

A = early_exit(5);
