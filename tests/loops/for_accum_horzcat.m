% Test: Horzcat accumulation with symbolic iteration count
% EXPECT: warnings = 1
% EXPECT_FIXPOINT: D = matrix[5 x (k+2)]

D = zeros(5, 2);
for i = 1:k
    D = [D, zeros(5, 1)];
end
