% Test: Variable first assigned inside loop body
% B is unbound pre-loop; bottom-as-identity means B gets the loop body's shape
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 3]
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[3 x 3]

for i = 1:n
    B = zeros(3, 3);
end
