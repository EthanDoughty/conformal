% Test: Return statement inside loop body
% Loop catches EarlyReturn, does post-loop join, doesn't propagate to caller
% EXPECT: warnings = 0
% EXPECT: A = matrix[1 x 1]

function y = loop_with_return(x)
    for i = 1:10
        if i > 5
            y = x;
            return;
        end
    end
    y = zeros(1, 1);
end

A = loop_with_return(zeros(3, 3));
