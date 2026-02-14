% Test: Recursive function (recursion guard prevents infinite loop)
% EXPECT: warnings = 1
% EXPECT: A = unknown

function y = factorial(n)
    y = n * factorial(n - 1);
end

A = factorial(5);
