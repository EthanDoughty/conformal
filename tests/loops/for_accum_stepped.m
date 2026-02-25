% Test: Accumulation with stepped range iteration count
% 1:2:10 has floor((10-1)/2)+1 = 5 iterations
% 1:3:12 has floor((12-1)/3)+1 = 4 iterations
% EXPECT: warnings = 2
% EXPECT_FIXPOINT: A = matrix[7 x 3]
% EXPECT_FIXPOINT: B = matrix[6 x 3]

A = zeros(2, 3);
for i = 1:2:10
    A = [A; zeros(1, 3)];
end

B = zeros(2, 3);
for j = 1:3:12
    B = [B; zeros(1, 3)];
end
