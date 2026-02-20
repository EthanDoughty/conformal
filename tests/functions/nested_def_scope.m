% Test: Nested function not visible at top level
% EXPECT: warnings = 1

function result = outer(n)
    result = inner(n);
    function y = inner(x)
        y = x + 1;
    end
end

% inner is not visible here: should emit W_UNKNOWN_FUNCTION
z = inner(5);
