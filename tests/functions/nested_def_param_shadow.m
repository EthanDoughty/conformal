% Test: Nested function parameter shadows parent variable (no write-back for params)
% EXPECT: warnings = 0
% EXPECT: result = matrix[4 x 4]

function result = outer(n)
    M = eye(n);
    y = inner(zeros(2, 2));
    result = M;
    function out = inner(M)
        out = M + M;
    end
end

result = outer(4);
