% Test: Iteration count extraction (concrete and symbolic)
% EXPECT: warnings = 2
% EXPECT: i = scalar
% EXPECT: j = scalar
% EXPECT_FIXPOINT: A = matrix[6 x 2]
% EXPECT_FIXPOINT: B = matrix[3 x (n+2)]

A = zeros(1, 2);
for i = 1:5
    A = [A; zeros(1, 2)];
end

B = zeros(3, 2);
for j = 1:n
    B = [B, zeros(3, 1)];
end
