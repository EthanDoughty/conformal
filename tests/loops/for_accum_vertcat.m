% Test: Vertcat accumulation with concrete iteration count
% EXPECT: warnings = 1
% EXPECT_FIXPOINT: C = matrix[13 x 4]

C = zeros(3, 4);
for i = 1:10
    C = [C; zeros(1, 4)];
end
