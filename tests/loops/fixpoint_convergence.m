% Test: Fixed-point convergence (scalar grows into matrix with unknown rows)
% Without --fixpoint: single-pass sees A change from scalar to matrix[2 x 1]
% With --fixpoint: join of scalar and matrix[2 x 1], matrix[3 x 1], etc. converges to matrix[None x 1]
% EXPECT: warnings = 0
% EXPECT: A = matrix[2 x 1]
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: A = unknown

A = 1;
for i = 1:3
    A = [A; 1];
end
