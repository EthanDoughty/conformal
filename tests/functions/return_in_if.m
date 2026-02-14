% Test: Return statement inside if-branch
% If then-branch returns early, result env is else-branch env
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

function y = conditional_return(x, cond)
    if cond
        y = x;
        return;
    else
        y = x + x;
    end
end

A = conditional_return(zeros(4, 4), 1);
