% Test: Post-loop join preserves pre-loop state (loop may not execute)
% Without --fixpoint: single-pass sees reassignment warning, result is matrix[3 x 3]
% With --fixpoint: join of pre-loop matrix[2 x 2] and post-body matrix[3 x 3] → matrix[None x None]
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]
% EXPECT_FIXPOINT: A = matrix[None x None]

A = zeros(2, 2);
for i = 1:n
    A = ones(3, 3);
end
