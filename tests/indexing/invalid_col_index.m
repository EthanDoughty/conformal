% Test 20: Constant column range indexing, no warnings are expected
% A is 4 x 6.
% c = A(:,2:5) -> 4 x 4
%
% EXPECT: warnings = 0
% EXPECT: c = matrix[4 x 4]

A = zeros(4, 6);
c = A(:, 2:5);