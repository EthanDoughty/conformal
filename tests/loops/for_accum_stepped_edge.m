% Test: Edge cases for stepped iteration counts
% 1:2:2 -> floor((2-1)/2)+1 = 1 iteration
% 1:2:1 -> floor((1-1)/2)+1 = 1 iteration
% 1:5:3 -> floor((3-1)/5)+1 = 1 iteration
% 10:-3:1 -> floor((10-1)/3)+1 = 4 iterations (negative step)
% EXPECT_FIXPOINT: D = matrix[3 x 3]
% EXPECT_FIXPOINT: E = matrix[3 x 3]
% EXPECT_FIXPOINT: F = matrix[3 x 3]
% EXPECT_FIXPOINT: G = matrix[6 x 3]

D = zeros(2, 3);
for i = 1:2:2
    D = [D; zeros(1, 3)];
end

E = zeros(2, 3);
for i = 1:2:1
    E = [E; zeros(1, 3)];
end

F = zeros(2, 3);
for i = 1:5:3
    F = [F; zeros(1, 3)];
end

G = zeros(2, 3);
for i = 10:-3:1
    G = [G; zeros(1, 3)];
end
