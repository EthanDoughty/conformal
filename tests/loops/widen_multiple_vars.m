% Test: Multiple variables with mixed stability in loop
% A grows (rows change), B stays constant
% EXPECT: warnings = 1
% EXPECT: A = matrix[4 x 2]
% EXPECT: B = matrix[4 x 4]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[(n+3) x 2]
% EXPECT_FIXPOINT: B = matrix[4 x 4]

A = zeros(3, 2);
B = ones(4, 4);
for i = 1:n
    A = [A; zeros(1, 2)];
    B = ones(4, 4);
end
