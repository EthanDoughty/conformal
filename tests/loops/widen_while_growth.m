% Test: While loop with matrix growth
% Same widening semantics as for loops
% EXPECT: warnings = 1
% EXPECT: A = matrix[2 x 5]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x 5]

A = zeros(1, 5);
while cond
    A = [A; ones(1, 5)];
end
