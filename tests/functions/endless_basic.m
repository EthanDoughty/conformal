% Test: calling end-less function from workspace
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 3]
A = zeros(3, 3);
B = endless_helper(A);
