% Test: Nested function writes back to parent workspace
% EXPECT: warnings = 1
% EXPECT: result = matrix[3 x 3]

function result = outer(n)
    M = eye(n);
    modify_m();
    result = M;
    function modify_m()
        M = zeros(3, 3);
    end
end

result = outer(5);
