% Test: Matrix growth with symbolic iteration count
% Without --fixpoint: single-pass sees A change from matrix[3 x 1] to matrix[4 x 1]
% With --fixpoint: join preserves column dimension, rows become None
% EXPECT: warnings = 1
% EXPECT: A = matrix[4 x 1]
% EXPECT_FIXPOINT: warnings = 3
% EXPECT_FIXPOINT: A = matrix[None x 1]

A = zeros(3, 1);
for i = 1:n
    A = [A; zeros(1, 1)];
end
