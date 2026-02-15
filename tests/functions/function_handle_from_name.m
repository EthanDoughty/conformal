% Test: Function handle from named function
% EXPECT: warnings = 0
% EXPECT: f = function_handle

function y = myFunc(x)
    y = x + 1;
end

f = @myFunc;
