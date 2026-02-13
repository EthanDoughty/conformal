% Test 19: Constant range indexing, no warnings are expected
% A is 3 x 6.
% r = A(2:3,:) -> 2 x 6
%
% EXPECT: warnings = 0
% EXPECT: r = matrix[2 x 6]

A = zeros(3, 6);
r = A(2:3, :);