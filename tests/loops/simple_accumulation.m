% Test: Single-pass misses feedback (only sees 1 iteration)
% Single-pass: sees A change from matrix[2 x 3] to matrix[(2+1) x 3]
% Emits reassignment warning due to incompatible shapes
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[7 x 3]

A = zeros(2, 3);
for i = 1:5
    A = [A; zeros(1, 3)];
end
