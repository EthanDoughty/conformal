% Test: Two variables growing on different axes in same loop
% A grows rows, B grows cols — each should widen independently
% EXPECT: warnings = 2
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT_FIXPOINT: warnings = 2
% EXPECT_FIXPOINT: A = matrix[(n+2) x 3]
% EXPECT_FIXPOINT: B = matrix[3 x (n+2)]

A = zeros(2, 3);
B = zeros(3, 2);
for i = 1:n
    A = [A; zeros(1, 3)];
    B = [B, zeros(3, 1)];
end
