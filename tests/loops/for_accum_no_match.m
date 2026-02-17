% Test: Patterns that should NOT trigger accumulation
% Self-ref delta: A = [A; A], stepped range, conditional accumulation
% EXPECT_FIXPOINT: F = matrix[None x 3]
% EXPECT_FIXPOINT: G = matrix[None x 3]
% EXPECT_FIXPOINT: H = matrix[None x 3]

F = zeros(2, 3);
for i = 1:n
    F = [F; F];
end

G = zeros(2, 3);
for i = 1:2:n
    G = [G; zeros(1, 3)];
end

H = zeros(2, 3);
for i = 1:n
    if i > 1
        H = [H; zeros(1, 3)];
    end
end
