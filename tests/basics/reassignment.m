% Test 12: Incompatible reassignment (warning expected)
% A is first assigned a 3 x 4 matrix, then reassigned a 2 x 2 matrix.
% The analysis should warn on definite shape incompatibility at reassignment.
%
% EXPECT: warnings = 1
% EXPECT: A = matrix[2 x 2]

A = zeros(3, 4);
A = zeros(2, 2);
