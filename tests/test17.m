% Test 17: Full slice and linear indexing, no warnings are expected
% A is 2 x 5.
% B = A(:,:) -> 2 x 5
% x = A(3)   -> scalar (conservative linear indexing)
%
% EXPECT: warnings = 0
% EXPECT: B = matrix[2 x 5]
% EXPECT: x = scalar

A = zeros(2, 5);
B = A(:, :);
x = A(3);