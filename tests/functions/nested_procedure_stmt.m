% Test: Nested procedure called as statement (no W_PROCEDURE_IN_EXPR).
% Also verifies W_PROCEDURE_IN_EXPR still fires when used as expression value.
% EXPECT: warnings = 1
% EXPECT: result = matrix[3 x 3]

function result = outer()
    M = eye(2);
    modify_m();
    result = M;
    bad = modify_m();  % EXPECT_WARNING: W_PROCEDURE_IN_EXPR

    function modify_m()
        M = zeros(3, 3);
    end
end

result = outer();
