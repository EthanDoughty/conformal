% Test: Nested function reads parent workspace variable
% EXPECT: warnings = 0
% EXPECT: result = matrix[4 x 4]

function result = outer(n)
    M = eye(n);
    result = get_m();
    function out = get_m()
        out = M;
    end
end

result = outer(4);
