% Test: Recursive function (recursion guard prevents infinite loop)
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = factorial(n)
    y = n * factorial(n - 1);  % EXPECT_WARNING: W_RECURSIVE_FUNCTION
end

A = factorial(5);
