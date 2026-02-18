% Test: end-less multi-function file (workspace call with subfunctions)
% EXPECT: warnings = 0
% EXPECT: B = matrix[2 x 2]
A = zeros(2, 2);
B = endless_multi_helper(A);
