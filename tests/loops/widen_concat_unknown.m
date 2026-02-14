% Test: Concat with unknown function result in loop body
% Unlike direct assignment, unknown wrapped in concat becomes
% matrix[None x None], so widening operates on two matrix shapes
% and produces a sound result.
% v0.11.0: join_dim made None absorbing (more sound) - concat with unknown
% loses column precision (matrix[None x None] instead of matrix[None x 3])
% EXPECT: warnings = 1
% EXPECT: A = matrix[None x None]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x None]

A = zeros(3, 3);
for i = 1:n
    A = [A; unknown_func()];
end
