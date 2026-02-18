% Test: space-separated multi-return function definition
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: C = matrix[2 x 2]
[A, B, C] = space_multi_return_helper(zeros(3, 3));
