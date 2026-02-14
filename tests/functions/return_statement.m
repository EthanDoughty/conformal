% Test: Return statement (early exit from function)
% EXPECT: warnings = 0
% EXPECT: A = scalar
% EXPECT: B = scalar

function result = early_exit(x)
    result = x;
    return;
end

function result2 = unreachable_after_return(x)
    result2 = x;
    return;
    result2 = zeros(100, 100);  % Not analyzed (dead code after return)
end

A = early_exit(5);
B = unreachable_after_return(10);
