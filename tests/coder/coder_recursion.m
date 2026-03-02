% Recursive function call in coder mode
% MODE: coder
% EXPECT: warnings >= 1
function y = coder_recursion(n)
    if n <= 1
        y = 1;
    else
        y = n * coder_recursion(n - 1);  % EXPECT_WARNING: W_CODER_RECURSION
    end
end
result = coder_recursion(5);
