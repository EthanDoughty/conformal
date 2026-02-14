% Test: Concat with unknown function result in loop body
% Unlike direct assignment, unknown wrapped in concat becomes
% matrix[None x None], so widening operates on two matrix shapes
% and produces a sound result.
% EXPECT: warnings = 1
% EXPECT: A = matrix[None x 3]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x 3]

A = zeros(3, 3);
for i = 1:n
    A = [A; unknown_func()];
end
