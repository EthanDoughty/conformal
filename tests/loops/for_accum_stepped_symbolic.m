% Test: Symbolic stepped range bails out (conservative)
% EXPECT: warnings = 1
% EXPECT_FIXPOINT: C = matrix[None x 3]

C = zeros(2, 3);
for i = 1:2:n
    C = [C; zeros(1, 3)];
end
